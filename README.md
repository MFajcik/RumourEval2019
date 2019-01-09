##Notes:
BERT:
predspracovanie cez twitter preprocesing (nahradenie url,mentionov atp specialnymi tokenmi) - pomohlo

pridanie stopwordov - pomohlo

pridanie featuru o source commente - pomohlo 0.837037 acc

pred tym:
Epoch 10, Validation loss|acc: 0.572339|0.826263 - (Best 0.5669|0.830976)
2019-01-04 16:12:39,089 DEBUG root: 0 - 0.76
1 - 0.75
2 - 0.89
3 - 0.93
4 - 0.95
5 - 0.94
6 - 0.94
7 - 0.97
8 - 0.87
9 - 1.00
10 - 0.86
11 - 1.00
12 - 1.00
13 - 1.00

potom 
Epoch 10, Validation loss|acc: 0.570702|0.837037 - (Best 0.5707|0.837037)


2019-01-07 14:06:01,456 DEBUG root: 0 - 0.71
1 - 0.77
2 - 0.91
3 - 0.93
4 - 0.96
5 - 0.93
6 - 0.94
7 - 0.97
8 - 0.87
9 - 1.00
10 - 0.86
11 - 1.00
12 - 1.00
13 - 1.00

Vysledky spred  rokov (iba na twitteri)
ECNU 	0.778 	
IITP 	0.641 	
kimber 	0.701 	
Mama Edha 	0.749 	
NileTMRG 	0.71 	
Turing 	0.769 	
Uwaterloo 	0.768 	


v jednom rune dokonca aj (hoci s este s trochu vyssou VL -- asi sa jedna iba o nahodu)
2019-01-07 14:41:51,189 INFO root: Epoch 5, Validation loss|acc: 0.580561|0.841751 - (Best 0.5806|0.841751)

uprava vstupu - rozdelenie src a prev postu

skusil som custom token \[p\] ale nezda sa ze by pomohol (acc 83,7710,83.0976) 
skusil som \[SEP\] token - nezda sa zeby nieco sposobilo (83.3670)
\[SEP\] token a 3 segmenty

uprava na batch 32 a dlzka viet 512 - 

pridanie klasifikacie veracity ku source tweetu - 



* Penalizacia u self attentionu nepomohla


* FIXME:
* dev samples in official baseline: 1485, dev samples in mine data - 1452 (4/1 - FIXED)

* labeling scheme
0 supp, 1 comm, 2 deny, 3 query

* distribucia hlbky datasetu

Dev data  
 Hlbka, Pocet  
('0', 38),  
('1', 810),  
('2', 211),  
('3', 122),  
('4', 102),  
('5', 67),  
('6', 52),  
('7', 31),  
('8', 23),  
('9', 15),  
('10', 7),  
('11', 4),  
('12', 2),  
('13', 1)  

Najlepsi vysledok Berta
2019-01-04 14:22:07,840 DEBUG root: Epoch 3, Validation loss|acc: 0.578753|0.836364 - (Best 0.5788|0.836364)
2019-01-04 14:22:07,840 DEBUG root: 0 - 0.79
1 - 0.77
2 - 0.90
3 - 0.93
4 - 0.94
5 - 0.94
6 - 0.92
7 - 0.94
8 - 0.87
9 - 0.93
10 - 0.86
11 - 1.00
12 - 1.00
13 - 1.00
# Ak dobre maskujem
* Baseline LSTM - najlepsie 83.2072

# TODO: recheck maskingu
* Baseline bez LSTM - najlepsie 83.5513
* Textovy model - najlepsie 80.5230557467309


* BERT: 83.6889

TODO: pocitat accuracy v zavilosti od hlbky v branchi

Napad na feature - hlbka v branchi? alebo

 staci source/ non source embedding?