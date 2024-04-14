-- creates all the tables and produces csv files
-- takes a while to run (about an hour)
SET search_path TO eicu_crd;
-- change the paths to those in your local computer using find and replace for '/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/eICU_data/'.
-- keep the file names the same

\i eICU_preprocessing/labels.sql
\i eICU_preprocessing/diagnoses.sql
\i eICU_preprocessing/flat_features.sql
\i eICU_preprocessing/timeseries.sql

-- we need to make sure that we have at least some form of time series for every patient in diagnoses, flat and labels
drop materialized view if exists ld_timeseries_patients cascade;
create materialized view ld_timeseries_patients as
  with repeats as (
    select distinct patientunitstayid
      from ld_timeserieslab
    union
    select distinct patientunitstayid
      from ld_timeseriesresp
    union
    select distinct patientunitstayid
      from ld_timeseriesnurse
    union
    select distinct patientunitstayid
      from ld_timeseriesperiodic
    union
    select distinct patientunitstayid
      from ld_timeseriesaperiodic)
  select distinct patientunitstayid
    from repeats;

\copy (select * from ld_labels as l where l.patientunitstayid in (select * from ld_timeseries_patients)) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/labels.csv' with csv header
\copy (select * from ld_diagnoses as d where d.patientunitstayid in (select * from ld_timeseries_patients)) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/diagnoses.csv' with csv header
\copy (select * from ld_flat as f where f.patientunitstayid in (select * from ld_timeseries_patients)) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/flat_features.csv' with csv header
\copy (select * from ld_timeserieslab) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/timeserieslab.csv' with csv header
\copy (select * from ld_timeseriesresp) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/timeseriesresp.csv' with csv header
\copy (select * from ld_timeseriesnurse) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/timeseriesnurse.csv' with csv header
\copy (select * from ld_timeseriesperiodic) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/timeseriesperiodic.csv' with csv header
\copy (select * from ld_timeseriesaperiodic) to 'D:/UKR/TPC-LoS-prediction/eICU_data_preprocessed/timeseriesaperiodic.csv' with csv header