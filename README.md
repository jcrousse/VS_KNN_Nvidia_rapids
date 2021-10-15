# VS-KNN: Pandas vs cuDF performance comparison

### How to run the code:
* get the `yoochoose-clicks.dat` public dataset.
* Set the path to the file as `data_sources.full_data` value in `config.json`
* run:
    * `python main.py` for the full end-to-end pipeline: Pre-processing, training and predict
    * use the `-s`, `-t` `-p` flags to skip the train/test split, train and predict respectively
    * use `-c` to use pandas instead of cudf and compare performances
    
### Default hyperparameters:  
* runs 100 predictions and calculate average and p90 prediction time 
* keeps the last `10` items in the current session
* looks up the last `5000` historical sessions per item
* keep the top `100` most similar session

### Results:
Approx 60x speed increase with **cuDF** over **pandas** on my RTX 2070
