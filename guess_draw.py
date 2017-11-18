import tensorflow as tf
import pygame
import numpy as np

#model variables
image_size = 28
conv1_filters = 20
dense1_nodes = 100
output_size = 10

red = (200,0,0)
white = (255,255,255)
black = (0,0,0)
stroke = 30
mag = 20
draw_size = image_size*mag
#functions
def getInput():
    array = np.zeros((image_size,image_size))
    for y in range(image_size):
        for x in range(image_size):
            sum = 0
            for j in range(mag):
                for i in range(mag):
                    sum += display.get_at((x*mag+i, y*mag+j))[0]*1.0
            average = sum/(mag*mag)
            array[y][x] = average
    return np.reshape(array, (1,image_size,image_size,1))

def printArray(array):
    for row in array:
        print(row)

#input layer
X = tf.placeholder(tf.float32,shape=(None, 28, 28, 1),name="X")

#Conv1
conv1 = tf.layers.conv2d(X,filters=conv1_filters,kernel_size=(5,5),padding="same",activation=tf.nn.relu,name="conv1")
pool1 = tf.layers.max_pooling2d(conv1,pool_size=[2,2],strides=2,padding="same",name="pool1")
#dense1
pool1_flat = tf.reshape(pool1,[-1,14*14*conv1_filters],name="pool1_flat")
dense1 = tf.layers.dense(pool1_flat,dense1_nodes,activation=tf.nn.sigmoid,name="dense1")

#output
op = tf.layers.dense(dense1,units=10,activation=tf.nn.sigmoid,name="op")
guess = tf.argmax(op, axis=1,name="guess")
#cost
Y = tf.placeholder(dtype=tf.int32,name="Y")
Y_one_hot = tf.one_hot(Y,10)
cost = tf.reduce_sum(-tf.multiply(Y_one_hot,tf.log(op))-tf.multiply(1-Y_one_hot,tf.log(1-op)),name="cost")


#save
saver=tf.train.Saver()

#run
with tf.Session() as sess:
    saver.restore(sess,"save/network.ckpt")
    #game loop
    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 100)
    display = pygame.display.set_mode((draw_size, draw_size))
    gameExit = False
    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    display.fill(black)
                if event.key == pygame.K_t:
                    guess_ = sess.run(guess, feed_dict={X:getInput()})
                    textSurface = myfont.render(str(guess_[0]), True, (255,255,255))
                    display.blit(textSurface,(0,0))
        if pygame.mouse.get_pressed()[0]:
            pygame.draw.ellipse(display, white, [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1], stroke,stroke])
        pygame.display.update()
    pygame.quit()
'''
Something intrinsically wrong with this.
The result is not improving over time.
The problem is adamOptimizer and the initializer
'''