
Part 1b:

1.
It is important that the features are on the same scale as that nearest neighbor relies on euclidean distance or else can cause incorrect classification. If though in real world problem features are encountered which are not on the same scale the data normalization is the way to resolve the issue. It brings the data in the same range hence making classification more accurate.
More Details:
https://stats.stackexchange.com/questions/287425/why-do-you-need-to-scale-data-in-knn

2.
Numerical Features are as the name suggest in the range of a maximum and minimum numbers for instance: Age 0-99 years (or more if genetics or healthy habits allow it :) )
On the other hand categorical features are more diverse it can be as simple as few categories feature like Gender or can be a bigger set like Animal kingdom classification (Carnivore, Herbivore and all that kind).

Categorical features can be represented in onhotfeature where it each possible value of a specific feature will be converted to a sub-feature and will be checked for each test data against those features. For instance: an animal can be a Carnivore (True or False) and can be a mammal (True or False) and so on.

3.
Testing Data helps in validating the model which has been trained to predict a certain label. It provides an understanding how good the model will perform in real world the testing data is hidden from the model when training.

4.
Supervised means that initially a set of data will be provided to the model to with the label/labels known which helps in building the relation between features and label.

5. After carefully looking at the dataset I noticed that the Iris versicolor and virginica are the two which are clustered together in graphical representation and if there were to exist a feature which can differentiate between the two that will further make classification more accurate. In a research it was found that height of the leaves along the flower-stem (cauline leaves) is a way to differentiate between the two considering virginica has longer leaves (extending above the flowers) while versicolor has shorter leaves (shorter or similar to height of the flowers).
Although, that raises my concern of how this will affect the classification with setosa, a possible solution is to introduce two layered classification where first just identifies setosa without considering the new added feature and the next layer includes the new feature to distinguish between the remaining two.
More details on the study:
http://www.clemson.edu/extension/hgic/water/resources_stormwater/rain_garden_plants_iris_versicolor.html

Part 2b

1.
Bag of words is good for textual analysis based of the frequencies of words in the text but it fails, since it breaks the whole text into words the contextual meaning is lost.

2.
All the features have equal weights as all of them are evaluated whether they’re present in a particular email or not. Due to onehotcoding.

3.
Yes, it did, since there’s a lot of room for improvement, for instance to begin with the data cleaning should help in improving the accuracy, presence of xml or html tags confuses the classifier. Secondly, the dataset is small so, that accounts for mis-classification as well.


