### What it does

Predicts tod and frog migration

### How to deploy a new version in production

Configure google cloud account and execute in root folder

```
gcloud beta functions deploy predict_toad_migration --memory=1024MB --runtime=python37 --trigger-http --max-instances 1 --source=production_code
```

### If yoh need Jupyter notebook with training experiments

start in this folder the docker image with Jupyter 
```
docker run -d -v /$(pwd)/:/home/jovyan/work -p 8888:8888 gaarv/jupyter-keras start-notebook.sh --NotebookApp.token=''
```
you may need to install additional libs
```
docker exec -it <container-id> pip install xxx yyy
```

and then type `localhost:8888` in your web browser and navigate to `amphibien.ipynb`