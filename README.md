# Efficient Redidual Dense Block Search

Get ready: 
    Extract the files in a folder ERDBS

    cd <path to ERDBS>
    mkdir Data parameters retrainedModel

Download Data from https://drive.google.com/drive/folders/1hd-C5iM11eHGcp6c-fY4fZC9-BoAHvEP?usp=sharing
Extract the Data zip file
Insert Div2k and benchmark folder to the Data folder created above

To evlove:
    python main.py -t evlove

    Some of the best models will be saved in parameters folder during evolution.
    Final Elites will be saved in a file named Elites.csv
    From the file Elites.csv and folder parameters good models can be selected and retrained

To train or retrain:
    python main.py -t train -s <sequence>

To test:
    python main.py -t test -s <sequence> -d <dataset>

sequence example : ssgcccgscggs
sequence consists s, g, and c representing SRDB, GRDB and CRDB respectively.
Length of the sequence must be >= 4 and <= 21.

datasets options : ['div2k', 'b100', 'set5', 'set14', 'urban100']