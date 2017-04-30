# README #

### [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) ###

* Quick summary

    Traffic signs are characterized by a wide variability in their visual appearance in real-world environments. For example, changes of illumination, varying weather conditions and partial occlusions impact the perception of road signs. In practice, a large number of different sign classes needs to be recognized with very high accuracy. Traffic signs have been designed to be easily readable for humans, who perform very well at this task. For computer systems, however, classifying traffic signs still seems to pose a challenging pattern recognition problem.

    We have explored different classifcation algorithms on shallow features(HoG, SIFT) to find a best classifier atleast to reach near human accuracy.

    Presentation [here](https://docs.google.com/presentation/d/1-WdLwKJzpL4ukYYuFl3g2TMUACUIbfdz0aWkfzQP7ZI/edit?usp=sharing)
    

* Version

  1.0.1
* [Content and Code](https://github.com/shravankumar147/gtsrb-smai/wiki)

  Please look at wiki pages to see code and results

### How do I get set up? ###

* Summary of set up

  We have first downloaded the dataset from [here](https://github.com/shravankumar147/gtsrb-smai/wiki/Database-Configuration), then understood the data through exploratory data analysis (EDA).
  
  Features Extracted on both training and testing dataset
  
  The Features are partitioned into training, testing and validating sets.
  
  [Different classification](https://github.com/shravankumar147/gtsrb-smai/wiki) algorithms applied on the features to find the best classifier.
  
  And tested on randomely chosen test dataset .
  
* Dependencies

   pip install -r requirements.txt
   
   We have used anaconda python.  
* Database configuration

  You can read More on dataset [here](https://github.com/shravankumar147/gtsrb-smai/wiki/Database-Configuration).
  Also links provided to download the training and testing datasets.
  
* How to run tests
  
  You can download the dependancies from requirements.txt. Once you have required packages, you can clone this repository and use self explanatory jupyter notebooks provided in src/ directory to run, which are using HoG descriptors as features. In addition if you want to use sift based features for you classification, you can use python scripts in src/siftbased directory, which also contained requirement.txt explicitely. In order to save the computation time, we have saved data, features, classifiers into respective folders, but due to their large size, we could not upload here. So, Once you run a code they will be stored into local directories, then they will be loaded back for every rerun within no time.


### Who do I talk to? ###

* Shravan Kumar
  CVIT,IIIT
  Shravankumar147@gmail.com
* Avijit Dasgupta
  CVIT,IIIT
  avijitdasgupta9@gmail.com
* JItendra Singh Chauhan
  jitendra.kec@gmail.com
* Nikhil Singh
  nikhils.iitk@gmail.com
