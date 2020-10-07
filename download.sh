#!/bin/sh

# Download data
wget https://s3.console.aws.amazon.com/s3/object/opiniondigest/data/yelp-default-data.zip
tar -zvxf yelp-default-data.zip
rm yelp-default-data.zip

# Download model files
wget https://opiniondigest.s3-us-west-2.amazonaws.com/model/yelp-default-model.tar.gz
tar -zvxf yelp-default-model.tar.gz
rm yelp-default-model.tar.gz

