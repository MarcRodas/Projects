import argparse
import os
import numpy as np
import cv2
from numpy.linalg import eig
from sklearn import neighbors
import scipy.linalg as scipylg

def create_data(path):
    x=[]
    y=[]
    for person in os.listdir(path):
        label = int(person[1:])
        for file in os.listdir(path + '/' + person):
            img = cv2.imread(path + '/' + person + '/' + file, 0)
            x.append(np.array(img).flatten())
            y.append(label)
    x = np.stack(x, axis=1)
    return x, y

def pca(train, test, dim):
    mean = np.mean(train, axis=0)
    train = train - mean
    test = test - mean
    A = train - mean
    C =  1/A.shape[1] * np.dot(A.T,A)

    eigval, eigvec = eig(C)
    eigvec = np.dot(A,eigvec)
    eigvec = eigvec/np.linalg.norm(eigvec, axis=0)
    eigval = (A.shape[0]/A.shape[1])*eigval

    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    eigvec = eigvec[:,:int(dim)]

    train = np.dot(train.T, eigvec)
    test = np.dot(test.T, eigvec)

    return train, test

def lda(x_train, y_train, test, dim):

    n_features = x_train.shape[1]
    class_labels = np.unique(y_train)
    
    mean = np.mean(x_train, axis=0)
    Sb = np.zeros((n_features,n_features))
    Sw = np.zeros((n_features,n_features))

    for l in class_labels:
        A_class = x_train[y_train == l]
        mean_class = A_class.mean(axis = 0)
        n_c = A_class.shape[0]

        Sb += n_c * (mean_class -mean) *(mean_class-mean).T
        Sw += np.dot((A_class-mean_class).T, (A_class-mean_class))
   
    A = np.linalg.inv(Sw).dot(Sb)
    eigval, eigvec = np.linalg.eigh(A)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    eigvec = eigvec[:,:int(dim)]

    x_train = np.dot(x_train, eigvec)
    test = np.dot(test, eigvec)

    return x_train, test

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PCA+LDA - FisherFaces')
    parser.add_argument('--train_dir', type=str, required=True, help='images train directory path')
    parser.add_argument('--test_dir', type=str, required=True, help='images test directory path')
    parser.add_argument('--d', type=float, required=False, default=149, help='number of data projection')
    parser.add_argument('--k', type=float, required=False, default=1, help='number of neighbours')

    args = parser.parse_args()

    x_train, y_train = create_data(args.train_dir)
    x_test, y_test = create_data(args.test_dir)
    x_train, x_test = pca(x_train, x_test, 200)
    x_train, x_test = lda(x_train,np.array(y_train), x_test, int(args.d))

    clf = neighbors.KNeighborsClassifier(n_neighbors=int(args.k))
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("Score: " + str(score))
