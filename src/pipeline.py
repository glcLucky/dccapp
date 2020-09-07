#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging.config
import logging
import datetime
import os
import datetime
import re
import os
import time
import click
import traceback
import logging.config
import logging
from tqdm import tqdm
from multiprocessing import Pool, Queue
from dotenv import load_dotenv, find_dotenv

from boto3plus import s3 as s3
from boto3plus.athena import AthenaAPI
from dcits.src.constants import cons
from . constants import cons
from . models.modelJay import etl_module
from . models.modelJay.predict_model import predicting_data
from . models.cell_rank import Cell_rank
from . visualization.run_notebooks_for_viz import run_notebook

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)
q = Queue()
temp_buffer = []
credentials_aws = {
    "access_key": os.environ.get("access_key"),
    "secret_key": os.environ.get("secret_key"),
    "region_name": os.environ.get("region_name")
}
s3aws = s3.load_resource_s3(credentials_aws)
athena = AthenaAPI(credaws=credentials_aws)

bucket_src = ''
bucket_tgt = ''
bucket_delivery = ''
MODEL_VERSION = 'v2'


def process_error(e):
    logger.error(e)


def write_to_queue(X_st_dt, X_end_dt, batch_size, n_sample_batch):
    """
    This function writes some messages to the queue.
    """


    # # check unique cells on prediction date

    check_date8_end = X_end_dt.strftime("%Y%m%d")
    check_date8_st = (X_end_dt.date() -
                    pd.Timedelta(days=cons.check_days-1)).strftime("%Y%m%d")

    query = ("SELECT DISTINCT({}) AS {} FROM {} WHERE "
            "(HOUR(date_time)=0) AND (class1 not in ('no_issue', 'normal')) "
            "AND date8 >= '{}' AND date8 <= '{}';".format(
                cons.name_cell_key,
                cons.name_cell_key,
                "prediction_results",
                check_date8_st, check_date8_end))

    df_temp_ = athena.query_as_dataframe(query)
    lst_cells_to_predict = df_temp_[cons.name_cell_key].unique().tolist()
    np.random.shuffle(lst_cells_to_predict)
    n_unique_cells = len(lst_cells_to_predict)
    logger.debug("{} cells to forecast".format(n_unique_cells))
    lst_cells_by_batch = []
    for i in range(0, n_unique_cells, batch_size):
        if i + batch_size <= n_unique_cells:
            lst_cells_by_batch.append(
                lst_cells_to_predict[i: i + batch_size])
        else:
            lst_cells_by_batch.append(lst_cells_to_predict[i:])

    if n_sample_batch != -1:
        lst_cells_by_batch = lst_cells_by_batch[:n_sample_batch]

    def add_to_queue(q, msg):
        """
        If queue is full, put the message in a temporary buffer.
        If the queue is not full, adding the message to the queue.
        If the buffer is not empty and that the message queue is not full,
        putting back messages from the buffer to the queue.
        """
        if q.full():
            temp_buffer.append(msg)
        else:
            q.put(msg)
            if len(temp_buffer) > 0:
                add_to_queue(temp_buffer.pop())

    for batch_id_, cells_ in enumerate(lst_cells_by_batch):
        logger.debug(batch_id_)
        msg_ = ";".join(cells_) + ";{};{};{}".format(
            batch_id_, X_st_dt, X_end_dt)
        add_to_queue(q, msg_)
    
    return len(lst_cells_by_batch)

def ts_pipeline(
    msg,
    weekday,
    forecast_id,
    forecast_table_name='raw_clean_data'):
    try_count = 1
    while True:
        logger.debug("Process {0} started, try_count={1}".format(
            os.getpid(), try_count))
        lst_cells_one_batch = msg.split(";")
        X_end_dt = pd.to_datetime(lst_cells_one_batch.pop())
        X_st_dt = pd.to_datetime(lst_cells_one_batch.pop())
        batch_id = lst_cells_one_batch.pop()

        logger.debug("retrieved, forecast={}, batch={}".format(
            forecast_id, batch_id))
        try:
            df_predict_yhat_, df_discarded_cells_  = forecast_data_by_batch(
                X_st_dt=X_st_dt,
                X_end_dt=X_end_dt,
                lst_cells_one_batch=lst_cells_one_batch,
                raw_table_name=forecast_table_name)
        except Exception as e:
            if try_count <= 5:
                try_count += 1
                time.sleep(60)
                continue
            else:
                logger.error(
                    "Error occured when predict, forecast={}, batch={}, "
                    "try_count={}, error={}...".format(
                        forecast_id, batch_id, try_count, e))
                traceback.print_exc()
                raise ValueError(e)

        logger.debug(
            "start to upload results to s3, forecast={}, batch={}...".format(
                forecast_id, batch_id))

        objectkey = (
            "short_term_forecast/batch_files/{}/{}/df_predict_yhat_batch_{}.csv.parquet.gzip"
            .format(weekday, forecast_id, batch_id))

        if s3.exist_object(bucket_tgt, objectkey):
            s3.delete_object(
                bucket=bucket_tgt, bucket_object=objectkey)
        s3.pandas_to_s3file(
            df_predict_yhat_, bucket_tgt, objectkey, compression="GZIP")

        objectkey = (
            "short_term_forecast/batch_files/{}/{}/df_discarded_cells_batch_{}.csv.parquet.gzip".
            format(weekday, forecast_id, batch_id))
        if s3.exist_object(bucket_tgt, objectkey):
            s3.delete_object(
                bucket=bucket_tgt, bucket_object=objectkey)
        s3.pandas_to_s3file(
            df_discarded_cells_, bucket_tgt, objectkey, compression="GZIP")

        logger.debug("upload results to s3 over, try_count={}...".format(
            try_count))
        break

def upload_data2s3(df_raw_clean, date, weekday, forecast_id):
    df_raw_clean = df_raw_clean.set_index(['date_time'], append=True)
    df_raw_clean = df_raw_clean.drop(columns=['date'])
    df_raw_clean = df_raw_clean.sort_index()

    logger.debug(
        "start upload raw_clean data to s3, date={}".format(date))
    objectkey = (
        "short_term_forecast/forecast_raw_clean_data/{}/{}/forecast_raw_clean_data_{}"
        ".csv.parquet.gzip".format(weekday, forecast_id, date))
    if s3.exist_object(bucket_tgt, objectkey):
        s3.delete_object(
            bucket=bucket_tgt, bucket_object=objectkey)
    s3.pandas_to_s3file(
        df_raw_clean, bucket_tgt, objectkey, compression="GZIP")

def dci_pipeline(
    date,
    weekday,
    forecast_id):
    from dci.src.pipeline import Pipeline as dci_Pipeline
    from dci.src.config import opt as dci_opt
    logger.debug(
        "start dci preiction, date={}, pid={}".format(date, os.getpid()))

    objectkey = (
        "short_term_forecast/forecast_raw_clean_data/{}/{}/forecast_raw_clean_data_{}"
        ".csv.parquet.gzip".format(weekday, forecast_id, date))

    if not s3.exist_object(bucket_tgt, objectkey):
        e = (
            "Unable to find the correponding forecast raw_clean_data in s3, "
            "weekday={}, forecast_id={}, date={}, objectkey={}".format(
                weekday, forecast_id, date, objectkey))
        logger.error(e)
        raise ValueError(e)

    df_raw_clean = s3.s3file_to_pandas(bucket_tgt, objectkey)

    logger.debug("start normalization, date={}".format(date))
    pipe = dci_Pipeline(opt=dci_opt)
    pipe.etl.df_raw_clean = df_raw_clean
    # normalization
    pipe.etl.run_normalizing_data()
    logger.debug("start prediction, date={}".format(date))
    # prediction
    pipe.prediction.run_predicting_data(df_raw_clean=pipe.etl.df_raw_clean,
                                        df_norm=pipe.etl.df_norm)
    logger.debug("start upload norm data to s3, date={}".format(date))
    # upload norm_data to s3
    objectkey = (
        "short_term_forecast/forecast_norm_data/{}/{}/forecast_norm_data_{}.csv.parquet.gzip"
        .format(weekday, forecast_id, date))
    if s3.exist_object(bucket_tgt, objectkey):
        s3.delete_object(
            bucket=bucket_tgt, bucket_object=objectkey)
    s3.pandas_to_s3file(
        pipe.etl.df_norm.copy(), bucket_tgt, objectkey, compression="GZIP")    

    logger.debug("start upload prediction data to s3, date={}".format(date))
    # upload pred_results to s3
    objectkey = (
        "short_term_forecast/forecast_dci_results/{}/{}/forecast_dci_results_{}.csv.parquet.gzip"
        .format(weekday, forecast_id, date))
    if s3.exist_object(bucket_tgt, objectkey):
        s3.delete_object(
            bucket=bucket_tgt, bucket_object=objectkey)
    s3.pandas_to_s3file(
        pipe.prediction.df_predict.copy(),
        bucket_tgt,
        objectkey,
        compression="GZIP")

    logger.debug(
        "start upload prediction client data to s3, date={}".format(date))
    # upload pred_results_client to s3
    objectkey = (
        "short_term_forecast/forecast_dci/{}/forecast_data_aws_ai.prediction"
        ".{}.gz".format(forecast_id, date))
    if s3.exist_object(bucket_delivery, objectkey):
        s3.delete_object(
            bucket=bucket_delivery, bucket_object=objectkey)
    s3.pandas_to_s3file(pipe.prediction.df_predict_client.copy(),
                        bucket_delivery,
                        objectkey,
                        index=False)
    logger.debug("dci pipeline over, date={}".format(date))


def priority_pipeline(
    forecast_st_date8,
    forecast_end_date8,
    X_end_date8,
    weekday,
    forecast_id):

    lst_cols = ['prefecture_name', 'bandwidth', 'carrier_id', 'sector_ref_id',
            'enb_number', 'enb_type', 'sector_id', 'carrier', 'branch_name',
            'eci', 'enb_name', 'cell_name', 'sector', 'enb_vendor',
            'date_time', 'cell_key', 'class1', 'class2', 'class3',
            'prob1', 'prob2', 'prob3']

    res_dci = s3.filter_object_bucket(
        bucket_tgt,
        "short_term_forecast/forecast_dci_results/{}/{}/forecast_dci_results_*".format(
            weekday, forecast_id))
    assert len(res_dci) == 7

    # avoid to use athena to query forecast dci results
    # because the glue service update the new data evey hour
    df_raw_forecast = pd.DataFrame()

    for dci_res_obj in tqdm(res_dci):
        df_raw_forecast_ = s3.s3file_to_pandas(bucket_tgt, dci_res_obj)
        df_raw_forecast_ = df_raw_forecast_.reset_index()
        df_raw_forecast_ = df_raw_forecast_[
            df_raw_forecast_.date_time.dt.hour == 0]
        df_raw_forecast_.columns = [
            col.lower() for col in df_raw_forecast_.columns]
        df_raw_forecast_ = df_raw_forecast_[lst_cols].reset_index(drop=True)
        df_raw_forecast = df_raw_forecast.append(df_raw_forecast_)
    
    df_raw_forecast = df_raw_forecast.reset_index(drop=True)
    df_raw_forecast['date_time'] = df_raw_forecast['date_time'].dt.strftime(
        "%Y-%m-%d %H %M %S")
    lst_cell_keys_forcast = df_raw_forecast['cell_key'].unique().tolist()

    table_history = "prediction_results"
    X_start_date_str = (
        pd.to_datetime(
            forecast_st_date8) - pd.Timedelta(days=7)).strftime("%Y%m%d")

    df_raw_history = athena.query_as_dataframe(
        query="SELECT {} FROM {} WHERE date8 between '{}' AND '{}' "
            "AND Hour(date_time) = 0 ORDER BY eci, date_time;".format(
                ",".join(lst_cols), table_history, X_start_date_str,
                X_end_date8))
    
    df_raw_history = df_raw_history.query("cell_key in {}".format(
        lst_cell_keys_forcast))

    cols = df_raw_forecast.columns
    df_raw_all = pd.concat(
        [df_raw_forecast, df_raw_history[cols]]).sort_values(
            ['cell_key', 'date_time']).reset_index(drop=True)
    lst_cols_nonidx = ['date_time', 'class1', 'class2', 'class3',
    'prob1', 'prob2', 'prob3', 'mse']
    lst_cols_idx = list(set(df_raw_all.columns) - set(lst_cols_nonidx))

    df_raw_all = df_raw_all.set_index(lst_cols_idx)
    cr = Cell_rank(df_raw_all, method='same_w', summary=False , prob=False)
    df_output = cr.df_d_cust_out

    logger.debug("start uploading cell_prioritization to cient s3")
    # upload cell_prioritization to cient s3
    objectkey = (
        "short_term_forecast/cell_prioritization/{}/cell_prioritization.{}.gz"
        .format(X_end_date8, X_end_date8))
    if s3.exist_object(bucket_delivery, objectkey):
        s3.delete_object(
            bucket=bucket_delivery, bucket_object=objectkey)
    s3.pandas_to_s3file(df_output, bucket_delivery, objectkey, index=False)

    logger.debug("start uploading cell_prioritization to internal s3")
    # upload cell_prioritization to internal s3
    df_output = df_output.rename(columns=cons.dic_jp2en_priority_output)
    objectkey = (
        "short_term_forecast/cell_prioritization/{}/cell_prioritization_{}.csv.parquet.gzip"
        .format(X_end_date8, X_end_date8))
    if s3.exist_object(bucket_tgt, objectkey):
        s3.delete_object(bucket=bucket_tgt, bucket_object=objectkey)
    s3.pandas_to_s3file(df_output, bucket_tgt, objectkey, compression="GZIP")
    

def pri_viz_pipeline(
    X_end_date8,
    kernel_name,
    bucket_name_src,
    bucket_name_tgt,
    bucket_name_delivery,
    weekday,
    forecast_id,
    forecast_st_date8,
    forecast_end_date8):
    logger.debug("start visualization for cell_prioritization client...")

    notebook_name = "priority_visualization.ipynb"
    if forecast_id.find('test') != -1:
        html_name = "priority_viz_{}_test.html".format(X_end_date8)
    else:
        html_name = "priority_viz_{}.html".format(X_end_date8)
    html_dir = run_notebook(
        kernel_name=kernel_name,
        notebook_name=notebook_name,
        html_name=html_name,
        bucket_name_src=bucket_name_src,
        bucket_name_tgt=bucket_name_tgt,
        bucket_name_delivery=bucket_name_delivery,
        forecast_id=forecast_id,
        forecast_st_date8=forecast_st_date8,
        forecast_end_date8=forecast_end_date8,
        weekday=weekday)
    
    s3.upload_object(
        html_dir,
        bucket=bucket_delivery,
        bucket_object='short_term_forecast/cell_prioritization/{}/{}'.format(
            X_end_date8, html_name))

    s3.upload_object(
        html_dir,
        bucket=bucket_tgt,
        bucket_object='short_term_forecast/cell_prioritization/{}/{}'.format(
            X_end_date8, html_name))
    os.remove(html_dir)

    logger.debug("start visualization for cell_prioritization internal...")

    notebook_name = "priority_visualization_internal.ipynb"
    if forecast_id.find('test') != -1:
        html_name = "priority_viz_internal_{}_test.html".format(X_end_date8)
    else:
        html_name = "priority_viz_internal_{}.html".format(X_end_date8)
    html_dir = run_notebook(
        kernel_name=kernel_name,
        notebook_name=notebook_name,
        html_name=html_name,
        bucket_name_src=bucket_name_src,
        bucket_name_tgt=bucket_name_tgt,
        bucket_name_delivery=bucket_name_delivery,
        forecast_id=forecast_id,
        forecast_st_date8=forecast_st_date8,
        forecast_end_date8=forecast_end_date8,
        weekday=weekday)

    s3.upload_object(
        html_dir,
        bucket=bucket_tgt,
        bucket_object='short_term_forecast/cell_prioritization/{}/{}'.format(
            X_end_date8, html_name))
    os.remove(html_dir)

    logger.debug("start to evaluate forecast performance...")

    notebook_name = "forecast_evaluation_v1.ipynb"
    if forecast_id.find('test') != -1:
        html_name = "forecast_evaluation_{}_test.html".format(X_end_date8)
    else:
        html_name = "forecast_evaluation_{}.html".format(X_end_date8)
    html_dir = run_notebook(
        kernel_name=kernel_name,
        notebook_name=notebook_name,
        html_name=html_name,
        bucket_name=bucket_name_tgt,
        forecast_id=forecast_id,
        weekday=weekday)

    s3.upload_object(
        html_dir,
        bucket=bucket_tgt,
        bucket_object='short_term_forecast/cell_prioritization/{}/{}'.format(
            X_end_date8, html_name))
    os.remove(html_dir)


class Pipeline():
    def __init__(self, *args, **kwargs):
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]
        
        # s3aws = s3.load_resource_s3(credentials_aws)
        # self.athena = AthenaAPI(credaws=credentials_aws)
        # self.bucket_src = s3aws.Bucket(self.s3_src_bucket)
        # self.bucket_tgt = s3aws.Bucket(self.s3_tgt_bucket)
        # self.bucket_delivery = s3aws.Bucket(self.s3_delivery_bucket)
        global bucket_src, bucket_tgt, bucket_delivery
        bucket_src = s3aws.Bucket(self.s3_src_bucket)
        bucket_tgt = s3aws.Bucket(self.s3_tgt_bucket)
        bucket_delivery = s3aws.Bucket(self.s3_delivery_bucket)
        if len(self.specified_date) == 0:
            self.latest_date = str(athena.query_as_dataframe(
                "SELECT MAX(date8) AS date8 FROM prediction_results;").iloc[0, 0])
        else:
            self.latest_date = self.specified_date

        self.X_end_dt = pd.to_datetime('{} 23:00:00'.format(self.latest_date))
        self.X_st_dt = self.X_end_dt - pd.to_timedelta(cons.n_behind_prefer-1, unit='H')
        self.forecast_st_date8 = (
            self.X_end_dt.date() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        self.forecast_end_date8 = (
            self.X_end_dt.date() + pd.Timedelta(
                days=cons.n_ahead/24)).strftime("%Y%m%d")
        self.X_end_date8 = self.X_end_dt.strftime("%Y%m%d")

        self.forecast_id = "{}_{}".format(
            self.forecast_st_date8, self.forecast_end_date8)
        if self.for_test:
            self.forecast_id = self.forecast_id + "_test"

    def start_wirte_to_queue(self):
        logger.debug(
            "Start writing messages to the queue, "
            "weekday={}, forecast_id={}".format(
                self.weekday, self.forecast_id))
        write_to_queue(
            X_st_dt=self.X_st_dt,
            X_end_dt=self.X_end_dt,
            batch_size=self.batch_size,
            n_sample_batch=self.n_sample_batch)
        logger.debug("End of writing message to queue")

    def start_ts_pipeline(self):
        logger.debug(
            "Start time-series forecast pipeline, "
            "weekday={}, forecast_id={}, n_batch={}, parent_process={}".format(
                self.weekday, self.forecast_id, q.qsize(), os.getpid()))

        if self.delete_existed_before_run:
            # delete existed files in s3 for this date
            logger.debug("start delete existed objects in buckets")
            for obj_ in s3.filter_object_bucket(
                bucket_tgt,
                regex="{}|cell_prioritization_{}".format(
                    self.forecast_id,
                    self.X_end_date8)):

                s3.delete_object(bucket=bucket_tgt, bucket_object=obj_)
        qsize = q.qsize()
        if self.max_n_processes > 1:
            logger.debug(
                "Use multiprocessing to forecase because of max_n_processes>1")
            pool = Pool(processes=self.max_n_processes,  maxtasksperchild=self.maxtasksperchild)
            for i in range(qsize):
                msg_ = q.get(block=True, timeout=None)
                pool.apply_async(ts_pipeline,
                                args=(
                                    msg_,
                                    self.weekday,
                                    self.forecast_id,
                                    self.forecast_table_name,),
                                error_callback=process_error)

            logger.debug("waiting all ts subprocessed done...")
            pool.close()
            pool.join()
            logger.debug("all ts subprocessed done...")
        else:
            logger.debug(
                "Use for loop to forecase because of max_n_processes=1")
            for i in range(qsize):
                msg_ = q.get(block=True, timeout=None)
                logger.debug("start process message: {}".format(msg_))
                ts_pipeline(
                    msg_,
                    self.weekday,
                    self.forecast_id,
                    self.forecast_table_name)

        logger.debug("start merge all batch files of forecast data")

        res_pred = s3.filter_object_bucket(
            bucket_tgt, "short_term_forecast/batch_files/{}/{}/df_predict*".format(
                self.weekday, self.forecast_id))

        if len(res_pred) == qsize:
            logger.debug("All batch files of pred_results found in s3")

            df_predict_yhat = pd.DataFrame()
            for i in tqdm(res_pred):
                df_predict_yhat_ = s3.s3file_to_pandas(bucket_tgt, i)
                if len(df_predict_yhat_) == 0:
                    continue
                df_predict_yhat = df_predict_yhat.append(df_predict_yhat_)
            if len(df_predict_yhat) == 0:
                e = "No any data in forecast data, forecast_id={}".format(
                        self.forecast_id)
                logger.error(e)
                raise ValueError(e)

            df_predict_yhat = df_predict_yhat.reset_index(['date_time'])
            df_predict_yhat['date'] = df_predict_yhat['date_time'].dt.strftime(
                "%Y%m%d")
            lst_unq_dates = sorted(df_predict_yhat['date'].unique().tolist())
            assert len(lst_unq_dates) == 7
            #import pdb; pdb.set_trace()
            logger.debug("End of merge, start split the data and upload by date")
            try:
                if self.max_n_processes > 1:
                    pool = Pool(processes=7)
                    for i in range(7):
                        date_ = lst_unq_dates[i]
                        pool.apply_async(
                            upload_data2s3,
                            args=(df_predict_yhat.query("date=='{}'".format(date_)),
                                date_,
                                self.weekday,
                                self.forecast_id),
                        error_callback=process_error)
                    logger.debug("waiting all dci subprocessed done...")
                    pool.close()
                    pool.join()
                else:
                    for i in range(7):
                        date_ = lst_unq_dates[i]
                        upload_data2s3(
                            df_predict_yhat.query("date=='{}'".format(date_)),
                            date_,
                            self.weekday,
                            self.forecast_id)
                
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                raise ValueError(e)
        else:
            e = ("Mismatch of number of batch files of forecast data "
                "between real and s3, real: {}, s3: {}".format(
                    qsize, len(res_pred)))
            logger.error(e)
            raise ValueError(e) 

        logger.debug("start merge all batch files of discarded data")
        res_discarded= s3.filter_object_bucket(
            bucket_tgt,
            "short_term_forecast/batch_files/{}/{}/df_discard*".format(
                self.weekday, self.forecast_id))

        if len(res_discarded) == qsize:
            logger.debug("All batch files of discarded data found in s3")
            df_discarded_cells = pd.DataFrame()
            for i in tqdm(res_discarded):
                df_discarded_cells_ = s3.s3file_to_pandas(bucket_tgt, i)
                df_discarded_cells = df_discarded_cells.append(df_discarded_cells_)
            objectkey = (
                "short_term_forecast/forecast_raw_clean_data/{}/{}/df_discarded_cells_{}"
                ".csv.parquet.gzip".format(
                    self.weekday, self.forecast_id, self.forecast_id))
            if s3.exist_object(bucket_tgt, objectkey):
                s3.delete_object(
                    bucket=bucket_tgt, bucket_object=objectkey)
            s3.pandas_to_s3file(
                df_discarded_cells, bucket_tgt, objectkey, compression="GZIP")
            logger.debug("combine discarded_cells data: over...")

        else:
            e = ("Mismatch of number of batch files of discarded data "
                "between real and s3, real: {}, s3: {}".format(
                    qsize, len(res_discarded)))
            logger.error(e)
            raise ValueError(e)
        logger.debug("End of ts pipeline")


    def start_dci_pipeline(self):
        logger.debug(
            "Start dci pipeline, "
            "weekday={}, forecast_id={}".format(
                self.weekday, self.forecast_id))
        lst_unq_dates = [
            k.strftime("%Y%m%d") for k in  pd.date_range(
                self.forecast_st_date8, self.forecast_end_date8)]
        lst_unq_dates = sorted(lst_unq_dates)
        assert len(lst_unq_dates) == 7
        if self.max_n_processes > 1:
            pool = Pool(processes=7)
            for i in range(7):
                date_ = lst_unq_dates[i]
                pool.apply_async(
                    dci_pipeline,
                    args=(date_,
                        self.weekday,
                        self.forecast_id),
                error_callback=process_error)
            logger.debug("waiting all dci subprocessed done...")
            pool.close()
            pool.join()
            logger.debug("End of ts pipeline")
        else:
            for i in range(7):
                date_ = lst_unq_dates[i]
                dci_pipeline(
                    date_,
                    self.weekday,
                    self.forecast_id)           


    def start_priority_pipeline(self):
        logger.debug(
            "Start priority pipeline, "
            "weekday={}, forecast_id={}".format(
                self.weekday, self.forecast_id))
        priority_pipeline(
            forecast_st_date8=self.forecast_st_date8,
            forecast_end_date8=self.forecast_end_date8,
            X_end_date8=self.X_end_date8,
            weekday=self.weekday,
            forecast_id=self.forecast_id,       
        )
        logger.debug("End of priority pipeline")

    def start_priority_viz_pipeline(self):
        logger.debug(
            "Start priority viz pipeline, "
            "weekday={}, forecast_id={}".format(
                self.weekday, self.forecast_id))
        pri_viz_pipeline(
            X_end_date8=self.X_end_date8,
            kernel_name=self.kernel_name,
            bucket_name_src=self.s3_src_bucket,
            bucket_name_tgt=self.s3_tgt_bucket,
            bucket_name_delivery=self.s3_delivery_bucket,
            weekday=self.weekday,
            forecast_st_date8=self.forecast_st_date8,
            forecast_end_date8=self.forecast_end_date8,
            forecast_id=self.forecast_id,
        )
        logger.debug("End of priority viz pipeline")


# # prediction pipeline by batch files

# ## load data

def forecast_data_by_batch(X_st_dt,
                           X_end_dt,
                           lst_cells_one_batch,
                           raw_table_name='raw_clean_data'):

    X_st_dt_query = X_st_dt - pd.to_timedelta(7*24, unit='H')  # for imputing

    query = ("SELECT * FROM {} WHERE (date_time BETWEEN timestamp '{}' "
             "AND timestamp '{}') AND {} in {};".format(
                 raw_table_name,
                 X_st_dt_query.strftime("%Y-%m-%d %H:%M:%S"),
                 X_end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                 cons.name_cell_key,
                 tuple(lst_cells_one_batch)))

    logger.debug("start to query data for this batch..., pid={}".format(os.getpid()))
    try:
        df_X_one_batch = athena.query_as_dataframe(query)
    except Exception as e:
        logger.error("Unable to query the data using athena, pid={}, error={}".format(os.getpid(), e))
        raise ValueError(e)

    logger.debug("query over, pid={}".format(os.getpid()))

    df_X_one_batch['date_time'] = pd.to_datetime(df_X_one_batch['date_time'])

    dict_cols_glue2std = dict(
        zip([k.lower() for k in cons.lst_std_cols], cons.lst_std_cols))
    df_X_one_batch = df_X_one_batch.rename(
        columns=dict_cols_glue2std).reindex(columns=cons.lst_std_cols)
    df_X_one_batch = df_X_one_batch.set_index(cons.lst_std_idx)

    ## imputing data across days

    etl_data = etl_module.etl_online(holidays=cons.dic_holidays,
                                     df_raw=df_X_one_batch)
    df_X_one_batch = etl_data.df_raw_imputed.copy()
    df_X_one_batch = df_X_one_batch.reset_index(level=['date_time'])
    df_X_one_batch = df_X_one_batch[df_X_one_batch['date_time'].between(
        X_st_dt, X_end_dt, inclusive=True)].copy()
    df_X_one_batch = df_X_one_batch.set_index(['date_time'], append=True)

    # ## cell-based model assignment
    logger.debug("cell-based model assignment...")
    lst_nbehind_available = [
        int(k.split("-")[0]) for k in cons.final_model_name[MODEL_VERSION][
            'normalday'].keys()]

    lst_n_behind_may = sorted(
        [n_behind_ for n_behind_ in lst_nbehind_available 
        if n_behind_ <= cons.n_behind_prefer])[::-1]

    df_X_one_batch_ = df_X_one_batch.reset_index().copy()
    dict_cells_by_model = {}

    for n_behind_ in lst_n_behind_may:
        df_X_one_batch_subset = df_X_one_batch_[
            df_X_one_batch_['date_time'].between(
            X_end_dt - pd.to_timedelta(n_behind_ - 1, unit='H'),
            X_end_dt)].copy()

        ts_size_ = df_X_one_batch_subset.groupby(['cell_key']).size()
        lst_cells_ = ts_size_[ts_size_ == n_behind_].index.tolist()
        dict_cells_by_model['{}-{}'.format(n_behind_, cons.n_ahead)] = lst_cells_
        df_X_one_batch_ = df_X_one_batch_.query(
            "cell_key not in {}".format(lst_cells_))
    dict_cells_by_model['discarded_cells'] = df_X_one_batch_[
        'cell_key'].unique().tolist()
    logger.debug("start to forecast data using trained model...")
    df_predict_yhat_all = pd.DataFrame()
    for k, v in dict_cells_by_model.items():
        logger.debug("{}: {}".format(k, len(v)))

    for k, v in dict_cells_by_model.items():
        if k == 'discarded_cells' or len(v) == 0:
            continue
        n_behind_ = int(k.split("-")[0])
        logger.debug("start forecast data using {} model, pid={}...".format(n_behind_, os.getpid()))
        df_in = df_X_one_batch[df_X_one_batch.index.isin(
            v, level=cons.name_cell_key)]
        df_in = df_in[df_in.index.isin(pd.date_range(
            X_end_dt - pd.to_timedelta(n_behind_ - 1, unit='H'),
            X_end_dt, freq='H'),
            level='date_time')]

        df_predict_yhat_ = predicting_data(opt="",
                                           df_in=df_in,
                                           n_behind=n_behind_,
                                           n_ahead=cons.n_ahead,
                                           dic_holidays=cons.dic_holidays)

        df_predict_yhat_all = df_predict_yhat_all.append(df_predict_yhat_)

    # format check
    if len(df_predict_yhat_all) != 0:
        df_predict_yhat_all = df_predict_yhat_all.reset_index()

        lst_lacked_vars = list(set(cons.lst_std_cols) 
                            - set(df_predict_yhat_all.columns))
        if len(lst_lacked_vars) > 0:
            e = ("Unable to find these required variables in predict results: "
                "{}".format(";".join(lst_lacked_vars)))
            logger.error(e)
            raise ValueError(e)
        lst_int_vars = ['bandwidth', 'carrier_ID', 'sector_ref_ID', 'eNB_number', 'sector_ID', 'ECI']
        df_predict_yhat_all[lst_int_vars] = df_predict_yhat_all[lst_int_vars].astype(int)
        df_predict_yhat_all = df_predict_yhat_all.reindex(
            columns=cons.lst_std_cols).set_index(cons.lst_std_idx)
        df_predict_yhat_all = df_predict_yhat_all.round(2) 
    else:
        df_predict_yhat_all = pd.DataFrame(
            columns=cons.lst_std_cols).set_index(cons.lst_std_idx)
  
    if dict_cells_by_model['discarded_cells'] != []:
        df_discarded_cells = df_X_one_batch[df_X_one_batch.index.isin(
            dict_cells_by_model['discarded_cells'], level='cell_key')].copy()
        df_discarded_cells['date'] = df_discarded_cells.reset_index()[
            'date_time'].dt.date.values
        df_discarded_cells = df_discarded_cells.groupby(
            level=['cell_key']).apply(lambda x: sorted(list(x['date'].unique())))
        df_discarded_cells = df_discarded_cells.to_frame(name="days_with_data")
        df_discarded_cells['history_data_st_date'] = X_st_dt
        df_discarded_cells['history_data_end_date'] = X_end_dt
    else:
        df_discarded_cells = pd.DataFrame()
    logger.debug("forecast over, pid={}".format(os.getpid()))
    return df_predict_yhat_all, df_discarded_cells
