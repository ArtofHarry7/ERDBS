# Efficient Redidual Dense Block Search

- Get ready: 
    Extract the files in a folder ERDBS

        cd  <path to ERDBS>
        mkdir Data parameters retrainedModel

    - Download [Data](https://drive.google.com/drive/folders/1hd-C5iM11eHGcp6c-fY4fZC9-BoAHvEP?usp=sharing).
    - Extract the zip file.
    - Insert Div2k and benchmark folder to the Data folder created above.

- To evlove:

        python main.py -t evlove

    - Some of the best models will be saved in parameters folder during the whole process of evolution.
    - Final Elites will be saved in a file named Elites.csv.
    - From the file Elites.csv and folder parameters good models can be selected and retrained

- To train or retrain:
    Train a unique sequence of model

        python main.py -t train -s <sequence>

- To test:
    Test a unique sequence of model on a given dataset

        python main.py -t test -s <sequence> -d <dataset>

    **Warniing : a model is only tested when it is alreasy trained else will return an error.**

    - sequence example : ssgcccgscggs
    - sequence consists s, g, and c representing SRDB, GRDB and CRDB respectively.
    - Length of the sequence must be >= 4 and <= 21.

    - datasets options : ['div2k', 'b100', 'set5', 'set14', 'urban100']