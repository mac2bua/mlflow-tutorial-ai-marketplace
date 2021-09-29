# MLflow Tutorial

This repository contains the materials used during the AI Marketplace 2021 MLflow tutorial.

Agenda
========

- ML Lifecycle                                                    
- Experiment Tracking                                       
- MLflow Models & Model Registry                   
- MLflow Projects                                              
- Advanced usage & tips


Running instructions
====================

In order to run the setup you will need to [install docker-compose](https://docs.docker.com/compose/install/) and then can simply do:
```shell
docker-compose build
docker-compose up -d
```

You can then access:
* Jupyter: `http://localhost:8888`
* MLfLow: `http://localhost:5000`
* minio (S3): `http://localhost:9000` (user: `minioadmin`, pass: `minioadmin`)
