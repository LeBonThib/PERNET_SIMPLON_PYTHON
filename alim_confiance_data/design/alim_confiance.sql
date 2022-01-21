CREATE TABLE raw_data(
   store_id VARCHAR(200),
   store_name VARCHAR(1000),
   store_siret VARCHAR(200),
   store_address VARCHAR(1000),
   store_zipcode VARCHAR(200),
   store_city VARCHAR(1000),
   store_industry VARCHAR(1000),
   store_approval VARCHAR(200),
   store_geoloc TEXT,
   store_filter VARCHAR(1000),
   store_industry_ods VARCHAR(1000),
   PRIMARY KEY(store_id)
);

CREATE TABLE training_data(
   store_id_training VARCHAR(200),
   store_siret_training VARCHAR(200),
   store_name_training VARCHAR(1000),
   store_address_training VARCHAR(1000),
   store_zipcode_training VARCHAR(200),
   store_city_training VARCHAR(1000),
   store_geoloc_training TEXT,
   store_filter_training VARCHAR(1000),
   PRIMARY KEY(store_id_training)
);

CREATE TABLE logs(
   prediction_id VARCHAR(200),
   prediction_log VARCHAR(200),
   store_id_training VARCHAR(200) NOT NULL,
   PRIMARY KEY(prediction_id),
   FOREIGN KEY(store_id_training) REFERENCES training_data(store_id_training)
);

CREATE TABLE inspection_data(
   inspection_id VARCHAR(200),
   inspection_date DATE,
   inspection_result VARCHAR(200),
   store_id VARCHAR(200) NOT NULL,
   prediction_id VARCHAR(200),
   PRIMARY KEY(inspection_id),
   UNIQUE(prediction_id),
   FOREIGN KEY(store_id) REFERENCES raw_data(store_id),
   FOREIGN KEY(prediction_id) REFERENCES logs(prediction_id)
);
