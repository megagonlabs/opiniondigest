#!/bin/sh

# Download data
wget https://opiniondigest.s3-us-west-2.amazonaws.com/data/yelp-default-data.tar.gz
tar -zvxf yelp-default-data.tar.gz
rm yelp-default-data.tar.gz

# Download model files
wget https://opiniondigest.s3-us-west-2.amazonaws.com/model/yelp-default-model.tar.gz
tar -zvxf yelp-default-model.tar.gz
rm yelp-default-model.tar.gz

