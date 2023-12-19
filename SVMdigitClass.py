import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


digits = datasets.load_digits()
#print(digits.images)
#print(digits.images.shape)
#print(digits.target)

images_and_labels = list(zip(digits.images, digits.target))
#print(images_and_labels)

#for index, (image, label) in enumerate (images_and_labels[:6]):
 #   plt.subplot(2,3, index+1)
  #  plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
   # plt.title('Target: %i'% label)
#plt.show()

data = digits.images.reshape((len(digits.images), -1))
#print(data)

classifier = svm.SVC(gamma=0.001)
train_test_split = int(len(digits.images) * 0.75)

classifier.fit(data[:train_test_split], digits.target[:train_test_split])

expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print('Confusion matrix: \n%s' % metrics.confusion_matrix(expected,predicted))
print(accuracy_score(expected,predicted))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print('predictions: ', classifier.predict(data[-2].reshape(1,-1)))
plt.show()


