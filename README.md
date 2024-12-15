### Intelligent Decision Support System  

### Planning of Los Angeles officers in anticipation of predicted crimes  

------------------------------

The **`Data`** folder contains:  
- The raw dataset named (`raw_dataset.csv`)  
- The dataset after preprocessing that we will use for training (`crime_dataset.csv`)  
- Validation data for the model (`validation_set_full.csv`)  

The **`Models`** folder contains:  
- A `.h5` file, which is the trained model.  
- A **`scripts`** folder with several Python files for preprocessing and the LSTM model.  

The **`Source`** folder contains:  
- Additional `.py` files for displaying statistics or processing data.  
- The **`ResourceAllocation`** folder with:  
  - Several `.py` files for resource allocation and generating random data for police officer locations.  
  - Several `.csv` files with different locations and police officer data.  
- The **`app`** folder with scripts for the website.  

------------------------------

#### Install dependencies:  

Run `install.sh`  
or you can simply do:  
`pip install -r requirements.txt`  

***Note:** This will install all dependencies; however, it is possible that some packages are missing, and you may need to install them manually.*  

------------------------------

#### Run the application:  

From the root of the project, you need to run:  
```sh  
streamlit run Source/app/Home.py  
```  

It will automatically open the application in your browser.  
If it doesn't happen, go to:  
[http://localhost:8501/](http://localhost:8501/) or [http://127.0.0.1:8501/](http://127.0.0.1:8501/)  

---
