import numpy as np
import scipy.special
import time
from matplotlib import pyplot as plt
import pygame
from scipy.ndimage import shift
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

class neuralNetwork:

    def __init__(self, lr, iNodes, hNodes, oNodes):

        self.count = 1

        #  defining learning rate
        self.learningRate = lr

        # making input_nodes, hidden_nodes, output_nodes
        self.iNodes = iNodes
        self.hNodes = hNodes
        self.oNodes = oNodes

        # making weights ( randomly intialized )
        self.wih = np.random.uniform(-1.0, 1.0, (self.hNodes, self.iNodes))
        self.who = np.random.uniform(-1.0, 1.0, (self.oNodes, self.hNodes))

        # defining activation function
        self.Sigmoid = lambda x: scipy.special.expit(x)

        # inverse of sigmoid
        self.inverse = lambda x: scipy.special.logit(x)

        self.count = 0

    def train(self, inputs, targets):

        self.count += 1
        print(self.count)

        #  forward prop
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.Sigmoid(hiddenInputs)

        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.Sigmoid(finalInputs)
        
        commanTerm = ( finalOutputs - targets ) * finalOutputs * ( 1 - finalOutputs )
       
        # updating weights
        self.who -= self.learningRate * ( commanTerm @ hiddenOutputs.T )
        self.wih -= self.learningRate * ((self.who.T @ commanTerm) * hiddenOutputs * (1 - hiddenOutputs)) @ inputs.T

        return self.who, self.wih

    def test(self, inputs, wih, who, inverse=False, inverseVector=None):

        if inverse is False:
            # assgning trained weights
            self.wih = wih
            self.who = who
            
            #  forward prop
            hiddenInputs = np.dot(self.wih, inputs)
            hiddenOutputs = self.Sigmoid(hiddenInputs)

            finalInputs = np.dot(self.who, hiddenOutputs)
            finalOutputs = self.Sigmoid(finalInputs)

            self.count += 1
            print(self.count)

            return finalOutputs

        else:
            inverseVector = np.array(inverseVector).reshape(10, 1).T
            I1 = self.inverse(inverseVector)
            I2 = np.dot(I1, self.who)

            # data rescaling (range 0.01-0.99)
            I2 = (( ( I2 - I2.min() ) / (I2.max() - I2.min()) ) * 0.98) + 0.01
    
            I3 = self.inverse(I2)
            I4 = np.dot(I3, self.wih)

            # data rescaling (range 0.01-0.99)
            I4 = (( ( I4 - I4.min() ) / (I4.max() - I4.min()) ) * 0.98) + 0.01
    
            image = I4.reshape((28, 28))
            image = gaussian_filter(image, sigma=0.84)
            plt.imshow(image, cmap="gray")
            plt.show()


if __name__ == "__main__":

    train_path = "D:\\Programming\\AI_developing\\Projects\\SSIP_Projects\\Handwritten Digit Recognize\\MNIST Dataset\\mnist_train.csv"
    test_path = "D:\\Programming\\AI_developing\\Projects\\SSIP_Projects\\Handwritten Digit Recognize\\MNIST Dataset\\mnist_test.csv"

    whoPath = "D:\\Programming\\AI_developing\\Projects\\Image_Recognizer_test\\who.npy"
    wihPath = "D:\\Programming\\AI_developing\\Projects\\Image_Recognizer_test\\wih.npy"

    # assining input_nodes, hidden_nodes, output_nodes, learning_rate and no. of iterations
    learningRate = 0.1
    input_nodes = 784
    hidden_nodes = 250
    output_nodes = 10
    ePoches = 1

    # making a instance of the class
    handwritten = neuralNetwork(learningRate, input_nodes, hidden_nodes, output_nodes)

    def train():

        # collecting training images to an array
        with open(train_path, "r") as file:
            data = file.readlines()
    
        # training the network
        for record in data:
            values = np.array(record.split(",")).astype(float)
            inputs = values[1:].reshape(784, 1)
            inputs = (inputs / 255.0 * 0.99) + 0.01
            targets = np.zeros((10, 1))
            targets[int(values[0])] = 0.99
            who, wih = handwritten.train(inputs, targets)

        # saving weights
        np.save(whoPath, who)
        np.save(wihPath, wih)
        print("weigths saved")
        time.sleep(5)

    def test():

        # loading weights
        wih = np.load(wihPath)
        who = np.load(whoPath)

        scoreCard = []

        # collecting test images to an array
        with open(test_path, "r") as file:
            data = file.readlines()

        # testing the network
        for record in data:

            # data preprocessing
            values = np.array(record.split(",")).astype(float)
            inputs = values[1:].reshape(784, 1)
            inputs = (inputs / 255.0 * 0.99) + 0.01

            # feeding image to the network
            outputs = handwritten.test(inputs, wih, who)

            # checking whether the prediction is correct
            if outputs.argmax() == values[0]:
                scoreCard.append(1)
            else:
                scoreCard.append(0)

        # getting the accuracy ( % )
        print(scoreCard.count(1) / len(scoreCard))

    # for ePoche in range(ePoches):
    #     print(f"ePoche no {ePoche}")
    # train()
    # test()

    def canvas():
        wih = np.load(wihPath)
        who = np.load(whoPath)

        pygame.init()
        screen = pygame.display.set_mode((280, 280))
        pygame.display.set_caption("Draw — S: Save, C: Clear")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

        def thicken_image(gray_img, size=2):
            structure = np.ones((size, size))
            thickened = grey_dilation(gray_img, footprint=structure)
            return np.clip(thickened, 0, 1)

        def center(image):
            cy, cx = center_of_mass(image)
            shift_y = np.round(image.shape[0]/2.0 - cy).astype(int)
            shift_x = np.round(image.shape[1]/2.0 - cx).astype(int)
            return shift(image, shift=[shift_y, shift_x], mode='constant', cval=0.0)

        drawing = False
        screen.fill((0, 0, 0))
        gray = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                if event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        screen.fill((0, 0, 0))
                    if event.key == pygame.K_s:
                        raw_pixels = screen.copy()
                        small = pygame.transform.scale(raw_pixels, (28, 28))
                        arr = pygame.surfarray.array3d(small)
                        arr = np.transpose(arr, (1, 0, 2))
                        gray = (np.dot(arr[..., :3], [0.299, 0.587, 0.114]) / 255.0 * 0.99) + 0.01
                        gray = center(gray)
                        gray = thicken_image(gray)
                        gray = gaussian_filter(gray, sigma=0.5)
                        inputs = gray.reshape(784, 1)
                        prediction = handwritten.test(inputs, wih, who)

                        confidance = int( ( prediction[ prediction.argmax() ] * 100 ) )
                        print(prediction.argmax())
                        Messagebox.show_info(f"Looks like a {prediction.argmax()}\nConfidance : {confidance}%", "Prediction")
                        
            else:
                if drawing:
                    x, y = pygame.mouse.get_pos()
                    pygame.draw.circle(screen, (255, 255, 255),(x,y),8)
                pygame.display.update()
                clock.tick(60)
                continue
            break

        pygame.quit()

    def reverse():
        wih = np.load(wihPath)
        who = np.load(whoPath)
        reverseArray = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        for i in range(0, 10):
            print(i)
            reverseArray[i] = 0.99
            handwritten.test(1, wih, who, True, reverseArray)
            reverseArray[i] = 0.01

    reverse()