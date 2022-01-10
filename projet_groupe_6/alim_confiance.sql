CREATE DATABASE alim_confiance;
USE alim_confiance;

CREATE TABLE raw_data(
   store_siret VARCHAR(50),
   store_name VARCHAR(50),
   store_address VARCHAR(255),
   store_zipcode VARCHAR(50),
   store_city VARCHAR(50),
   store_industry VARCHAR(50),
   store_approval VARCHAR(50),
   store_geoloc TEXT,
   store_filter VARCHAR(50),
   store_industry_ods VARCHAR(50),
   PRIMARY KEY(store_siret)
)ENGINE=InnoDB DEFAULT CHARSET=utf8MB4;

CREATE TABLE training_data(
   store_siret_training VARCHAR(50),
   store_name_training VARCHAR(50),
   store_address_training VARCHAR(255),
   store_zipcode_training VARCHAR(50),
   store_city_training VARCHAR(50),
   store_geoloc_training TEXT,
   store_filter_training VARCHAR(50),
   PRIMARY KEY(store_siret_training)
);

CREATE TABLE logs(
   prediction_id VARCHAR(50),
   prediction_log VARCHAR(50),
   store_siret_training VARCHAR(50) NOT NULL,
   PRIMARY KEY(prediction_id),
   FOREIGN KEY(store_siret_training) REFERENCES training_data(store_siret_training)
);

CREATE TABLE inspection_data(
   inspection_id VARCHAR(50),
   inspection_date DATE,
   inspection_result VARCHAR(50),
   store_siret VARCHAR(50) NOT NULL,
   prediction_id VARCHAR(50),
   PRIMARY KEY(inspection_id),
   UNIQUE(prediction_id),
   FOREIGN KEY(store_siret) REFERENCES raw_data(store_siret),
   FOREIGN KEY(prediction_id) REFERENCES logs(prediction_id)
);
