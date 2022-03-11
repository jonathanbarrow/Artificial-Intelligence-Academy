gt5fr2+

..................+333333333333333333333333333###### AIA Foundations class notes

############################# Class and Objects

import datetime as dt

class Person():
    "Creates a Person Object for our program"
    def __init__(self, fname, lname, dob):
        self.fname = fname
        self.lname = lname
        self.dob = dob

    def get_name(self):
        return "{0} {1}".format(self.fname, self.lname)

    def calc_age(self):
        today = dt.datetime.now()
        diff = today - dob
        days = diff.days
        years = days / 365
        return int(years)

    def __str__(self):
        return "Person: [fname={0}, lname={1}]".format(self.fname, self.lname)

class Employee(Person):
    def __init__(self, fname, lname, dob, title):
        Person.__init__(self, fname, lname, dob)
        self.title = title

    def __str__ (self):
        line = "Employee: [fname={0}, lname={1}], title = {2}"
        return line.format(self.fname, self.lname, self.title)

if __name__ == '__main__':
    # Create Person
    dob = dt.datetime(1984, 11, 22)
    p = Person("Adam", "Gaweda", dob )
    k = Person("Whosist", "Whatyurface", dt.datetime.now())
    e = Employee("Adam", "Gaweda", dob, "Instructor")
    print(p)

# Turtle is a library 

class TV():
  def __init__(self):
    self.channel = 1
    self.volume = 10
    self.on = False

  def turn_on(self):
    self.on = True

  def turn_off(self):
    self.on = False

  def increase_volume(self):
    self.volume += 1

  def decrease_volume(self):
    self.volume -= 1

  def channel_up(self):
    # Only 120 channels
    self.channel = (self.channel + 1) % 120

  def channel_down(self):
    # Only 120 channels
    self.channel = (self.channel - 1) % 120

  def set_channel(self, new_channel):
    if new_channel > 0 and new_channel <= 120:
      self.channel = new_channel
    else:
      print("Unknown Channel")

"""if __name__ == '__main__':
  tv = TV()
  for i in range(10):
    tv.increase_volume()

  tv.set_channel(300)

  print("Current Volume:", tv.volume)
  print("Current Channel:", tv.channel)"""



######################### File Handling, Dictionary, API's.

## For gaining access to readings.txt This is absolute files paths

filename = "C:/Users/Jonathan/Desktop/IO_Demo/data/readings.txt"
# Swap slashes
fi = open(filename, "r")

contents = fi.read() # Reads all lines
contents = fi.readlines()

total = 0
for line in contents:
  break_down = line.split()
  temp = break_down[3]
  temp = int(temp)
  total = total + temp
  #break 

avg = total / len(contents)
avg = avg / 10
print(total / len(contents))

fi.close()

## Relative File Paths

filename = "../data/readings.txt" # This would work on all machines as 
# long as they have a folder called data and readings.txt inside. 


####################################### Scientific Libraries


###############Scientific Modules

import numpy as np

x = np.array([1, 2, 3])
y = [4, 5, 6]

print(x + y)

z = np.array([[1,2,3],[4, 5, 6], [7,8,9]] )
w = np.array([[1,2], [3,4], [5,6]])

dot = np.dot(z, w)

print(dot)

import csv

filename = "C:/Users/jonat/OneDrive/Documents/iris.csv"
fi = open(filename, 'r')
reader = csv.DictReader(fi)

for field in reader.fieldnames:
  print(field)

fi.close()


################### Polynomials

def dot_product(x, y):
    dot = 0
    for a, b in zip(x,y):
        dot += a * b
    return dot


####################################### Data Analysis

################### Matplot lib

import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [5,6,7,8])

plt.ylabel("Frequency")
plt.yticks([5, 6, 7, 8])
plt.title("Title to my Plot")
plt.show()

################# Subplots

import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [5,6,7,8]

fig, ax = plt.subplots(2,3, sharex=True, sharey=True )

ax[0,0].plot(x,y)
ax[0,2].scatter(x,y)
ax[0,2].set_xlabel("Hello")
ax[1,1].plot(x,y)
ax[1,1].scatter(x,y)
plt.show()

################# Example of Using Subplots

from email import header
import matplotlib.pyplot as plt
import csv

fig, axes = plt.subplots(4,4, sharex=True, sharey=True)

for i in range(4):
    for j in range(4):
        setosa_x_axis = []
        setosa_y_axis = []
        versicolor_x_axis = []
        versicolor_y_axis = []
        virginica_x_axis = []
        virginica_y_axis = []
        with open("C:/Users/jonat/OneDrive/Documents/iris.csv", 'r') as contents:
            # Skip header
            header = contents.readline()
            reader = csv.reader(contents)
            for line in reader:
                sepal_length = line[0]
                sepal_width = line[1]
                petal_len = line[2]
                petal_width = line[3]
                species = line[4]
                # Recall how lists operate a little different than data types like ints
                # or floats. Below creates a reference point to which list we want to 
                # append to.
                if species == "setosa":
                    x_axis = setosa_x_axis
                    y_axis = setosa_y_axis
                elif species == "versicolor":
                    x_axis = versicolor_x_axis
                    y_axis = versicolor_y_axis
                else:
                    x_axis = virginica_x_axis
                    y_axis = virginica_y_axis

                # Mapping to easily grab the specific data point we want to work 
                # with for this particular row/col combination
                # For example, 0,0 is sepal_len/sepal_width; 2, 1 is petal_len/sepal_width
                mapping  = {0: sepal_length, 1: sepal_width, 2: petal_len, 3:petal_width }
                x_axis.append(mapping[i])
                y_axis.append(mapping[j])

            msize = 20 # Increase marker size
            axes[i, j].scatter(setosa_x_axis, setosa_y_axis, label="setosa", c="red", s=msize)
            axes[i, j].scatter(versicolor_x_axis, versicolor_y_axis, label="versicolor", c="green", s=msize)
            axes[i, j].scatter(virginica_x_axis, virginica_y_axis, label="virginica", c="blue", s=msize)

            # Hide the ticks
            axes[i, j].get_xaxis().set_ticks([])
            axes[i, j].get_yaxis().set_ticks([])

            # Similar to mapping, grab the specific labels for the x/y axes
            # depending on the row and column we are on in the for loops
            labels = {0: "sepal_len", 1: "sepal_width", 2: "petal_len", 3:"petal_width"}
            axes[i, j].set_xlabel(labels[i])
            axes[i, j].set_ylabel(labels[j])

# place legend in bottom right corner
plt.legend(loc='best', bbox_to_anchor=(1,0,1,0))
plt.show()

################# Example of Using Subplots/PAAAAAAAAAAAAANDAS

import pandas as pd


df = pd.read_csv("C:/Users/jonat/OneDrive/Documents/iris.csv")
print(df)

print(df['species'] == 'setosa')

setosa = df[df['species'] == 'setosa']
print(setosa)

species = df.groupby('species')
for name, data in species:
    print(name)
    print(data['sepal_length'].mean())


import pandas as pd
import matplotlib.pyplot as plt

filename = "../data/iris.csv"
df = pd.read_csv(filename)

x = "sepal_length"
y = "petal_width"
species = df.groupby("species")
for name, data in species:
  plt.scatter(data[x], data[y], label=name)

plt.legend()
plt.show()


import pandas as pd

df = {'length': [1,  2,  3,  4],
      'width':  [5,  6,  7,  8],
      'depth':  [9, 10, 11, 12]}

df = pd.DataFrame(df)
df.to_csv("new_csv_file.csv")


################################################# SQL

######################################## Connecting to SQLite
import sqlite3

filename = '../data/Chinook_sqlite.sqlite'
conn = sqlite3.connect(filename)
cursor = conn.cursor()

query = """SELECT * 
FROM Track
WHERE UnitPrice > 0.99
"""
result = cursor.execute(query)

########################################## Advanced Queries and Using Pandas
import sqlite3

filename = '../Chinook_sqlite.sqlite'
conn = sqlite3.connect(filename)
cursor = conn.cursor()

df = pd.read_sql_query(query, conn)
df.head()

conn = sqlite3.connect("Chinook_Sqlite.sqlite")
query = """SELECT Track.Name, Album.Title, Artist.Name, COUNT(InvoiceID)
as NumPurchases
FROM Track
INNER JOIN Album ON Album.AlbumID = Track.AlbumID
INNER JOIN Artist ON Artists.ArtistID = Album.ArtistID
INNER JOIN InvoiceLine ON InvoiceLine.TrackID = Track.TrackID
GROUP BY InvoiceID
"""

df = pd.read_sql_query(query, conn)