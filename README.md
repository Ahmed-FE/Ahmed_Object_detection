# CMMI_Task
- Run task 1 to generate the json files required in task 1
- run task 2 to generate the josn files required for task 2
- run task three to generate the videos required in task 3
- the model uses python 3.10 with deep sort real time , Torch ,open cv , ultralytics and date time . You need to install all of them latest versions . for instant to install ultralytics run in the terminal 
  pip install ultralytics
  pip install opencv-python
  pip pip install deep-sort-realtim
- when using Midas just run the code and it will ask you for any missing dependencies.
- when running the code just change the path to where your data is located and change from north to south to indicate which file you are using .
- in the code there is some commented line of codes if you want to generate specific things indicated in the code for instance in the helper files in the function distance_for_detected_object_in_a_frame
there are many commented lines to show images and the line of code that call that function in task 2 but I don't recommend calling it while the while loop is running . the model is heavy enough already . it might not complete even with using GPU .
- Helper functions contain all the helper functions it has to be located in the same file with the tasks files .
- if anything is unclear let me know 
