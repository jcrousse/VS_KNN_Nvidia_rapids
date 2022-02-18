# VS-KNN: Pandas vs cuDF performance comparison

### How to run the code:
* Get the [RSC15 "yoochoose"](https://www.kaggle.com/chadgostopp/recsys-challenge-2015) dataset, and copy it in the 
`archive folder`
* Run a docker container with the RAPIDS suite. For example:  
```shell
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786  -v /path/to/this/repo:/rapids/notebooks/host  rapidsai/rapidsai:cuda11.2-runtime-ubuntu18.04-py3.7 
```
* go to `localhost:8888` and run the `demo.ipynb` notebook !

   
### Default hyper parameters:  
* runs 100 predictions and calculate average and p90 prediction time 
* keeps the last `10` items in the current session
* looks up the last `5000` historical sessions per item
* keep the top `100` most similar session

### Results:
Approx 60x speed increase with **cuDF** over **pandas** on my RTX 2070.

