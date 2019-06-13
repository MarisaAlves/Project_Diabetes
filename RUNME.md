# Running Lasso Regression File

## How to Run the Python Files

<p>To run the Lasso_Regression.py file (or any of the python files for this project) one must first ensure the computer has all of the necessary python libraries installed on their system. The libraries needed are matplotlib, pandas, os, sklearn, numpy, math, sklearn, scipy and seaborn.<p>
<p>To install these libraries, you will use the pip installer in the terminal and run the code as follows:

```python
pip3 install matplotlib
```

### To Run the Code Once Libraries Installed
For running the code one has 2 options:
1. Run it within the terminal
2. Run it in an editor such as PyCharm

### Running in the Terminal
<p>To run the python file in the terminal, it is important to ensure you are in the proper directory containing the Lasso_Regression.py file, as well as the diabetes.data. To run, you would then perform the following command:
```python
python3 diabetes_regression.py
```

### Running in PyCharm
<p>To run the python file in PyCharm you must first create a PyCharm project. This project will create all the project metadata to help run the file, and the project folder must contain the Lasso_Regression.py file, as well as the diabetes.data file. To run the python file, either click on the green arrow at the top right of the screen, or right click within the body of the code and then click on run. 

## How to Run Using the DockerFile

<p>Alternatively, as opposed to installing the libraries and ensuring the computer has the proper components, you could run the DockerFile.<p>
1. The first thing needed to accomplish this is to download the Lasso_Regression.py file, as well as the diabetes.data file.
2. Ensure the DockerFile has been added to the same directory containing the files above (they will all be packaged into the same folder inside the Dockerfile). In the terminal, ensure you are at the directory level containing the DockerFile.
3. Then we must build the DockerFile by running the following command in the terminal:
```
docker build -t docker_diabetes .
```
The name following the -t will be the new assigned name of the file. It could be anything you'd like, but should represent what the file does, and must be all lowercase. It is very important to not forget the . at the end, as it indicates the Dockerfile is in the current directory.
4. Now we run our built file by performing the following command in the terminal:
```
docker run docker_diabetes
```
<p>



