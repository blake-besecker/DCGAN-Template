import tensorflow as tf
import os
from matplotlib import pyplot as plt
BATCH_SIZE = 32
EPOCHS = 100
noise_dim = 100
examples = 5
#import dataset and normalize data
dataset = tf.keras.utils.image_dataset_from_directory('/Users/blakebesecker/Downloads/data', image_size=(128,128), batch_size=BATCH_SIZE, label_mode=None, shuffle=True)
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
normalizer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

dataset = dataset.map(lambda x: normalizer(x))
def Generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((8*8*512),use_bias=False,input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #first 'stack' ends, second begins
    model.add(tf.keras.layers.Reshape((8,8,512)))
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(2,2),padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2),padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2),padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    
    model.add(tf.keras.layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

def Discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5,5), strides=(2,2),padding='same', use_bias=False,input_shape=[128, 128, 3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (5,5), strides=(2,2),padding='same', use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model
#define loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator = Generator()
discriminator = Discriminator()

#save some checkpoints 
checkpoint_dir = './anime_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)





@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    #begin training, signalled by using the tape
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images= generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    dis_gradients = dis_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(dis_gradients,discriminator.trainable_variables))
    print('.')
    
    
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('!')

      
def generate_and_show_image(generator, noise_dim):
    # Create random noise vector
    noise = tf.random.normal([1, noise_dim])
    
    # Generate image from noise
    generated_image = generator(noise, training=False)
    
    # Remove batch dimension and rescale pixel values from [-1,1] to [0,1]
    img = (generated_image[0].numpy() + 1) / 2.0
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
train(dataset, EPOCHS)

generate_and_show_image(generator, noise_dim)
generate_and_show_image(generator, noise_dim)
generate_and_show_image(generator, noise_dim)