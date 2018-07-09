from __future__ import print_function
import os,timeit
import PIL.Image as Image
import numpy as np 
import theano
import theano.tensor as T 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images,shape_data

class RBM():
    '''Restricted Boltzmann Machine (RBM)  '''
    def __init__(self,input=None,n_visible=784,n_hidden=500,W=None,hbias=None,vbias=None,numpy_rng=None,theano_rng=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if numpy_rng == None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if W == None:
            initial_W = np.asarray(numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)),
                    dtype=theano.config.floatX)
            W = theano.shared(value=initial_W,name='W',borrow=True)
        if hbias == None:
            hbias = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX),name='hbias',borrow=True)
        if vbias == None:
            vbias = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX),name='vbias',borrow=True)
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.params = [self.W,self.hbias,self.vbias]

    def free_energy(self,v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -1 * (hidden_term + vbias_term)
        
    def propup(self,vis):
        '''propagates the visible units activation upwards t0 the hidden units.'''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self,v0_sample):
        '''infers state of hidden units given visible units.'''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,n=1, p=h1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self,hid):
        '''propagates the hidden units activation downwards to the visible units.'''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self,h0_sample):
        '''infers state of visible units given hidden units.'''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        '''implements one step of Gibbs sampling,starting from the hidden state.'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        '''implements one step of Gibbs sampling,starting from the visible state.'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        '''implements one step of CD-k or PCD-k'''
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
            
        ([pre_sigmoid_nvs,nv_means,nv_samples,pre_sigmoid_nhs,nh_means,nh_samples],updates) = \
        theano.scan(self.gibbs_hvh,outputs_info=[None, None, None, None, None, chain_start],n_steps=k,name="gibbs_hvh")
       
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr,dtype=theano.config.floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        '''Stochastic approximation to the pseudo-likelihood'''
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        '''Approximation to the reconstruction error'''
        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),axis=1))
        return cross_entropy

    def train(self,data=None,learning_rate=0.1, training_epochs=15,batch_size=20, output_folder='rbm_plots'):
        '''train and afterwards sample from it using Theano.'''
        n_train_batches = data.get_value(borrow=True).shape[0]//batch_size

        index = T.lscalar()   
        x = T.matrix('x')
        self.input = x
        persistent_chain = theano.shared(np.zeros((batch_size, self.n_hidden),
                                                    dtype=theano.config.floatX),borrow=True)
        cost, updates = self.get_cost_updates(lr=learning_rate,
                                            persistent=persistent_chain, k=15)
        # if not os.path.isdir(output_folder):
        #     os.makedirs(output_folder)
        # os.chdir(output_folder)
        train_rbm = theano.function([index],cost,updates=updates,givens={
                x: data[index * batch_size: (index + 1) * batch_size]},name='train_rbm')

        plotting_time = 0.
        start_time = timeit.default_timer()
        for epoch in range(training_epochs):
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost += [train_rbm(batch_index)]
            print('Training epoch %d, cost is ' % epoch, np.mean(mean_cost))

            # plotting_start = timeit.default_timer()
            # image = Image.fromarray(tile_raster_images(X=self.W.get_value(borrow=True).T,img_shape=(28, 28),tile_shape=(10, 10),tile_spacing=(1, 1)))
            # image.save('filters_at_epoch_%i.png' % epoch)
            # plotting_stop = timeit.default_timer()
            # plotting_time += (plotting_stop - plotting_start)

        end_time = timeit.default_timer()
        pretraining_time = (end_time - start_time) - plotting_time
        print ('Training took %f minutes' % (pretraining_time / 60.))
       
    def sample(self,input=None,n_chains=20,n_samples=10):
        number_of_test_samples = input.get_value(borrow=True).shape[0]
        test_idx = self.numpy_rng.randint(number_of_test_samples - n_chains)
        persistent_vis_chain = theano.shared(np.asarray(
                input.get_value(borrow=True)[test_idx:test_idx + n_chains],
                dtype=theano.config.floatX))
    
        plot_every = 1000
        ([presig_hids,hid_mfs,hid_samples,presig_vis,vis_mfs,vis_samples],updates) = \
        theano.scan(self.gibbs_vhv,outputs_info=[None, None, None, None, None, persistent_vis_chain],n_steps=plot_every,name="gibbs_vhv")

        updates.update({persistent_vis_chain: vis_samples[-1]})
        sample_fn = theano.function([],[vis_mfs[-1],vis_samples[-1]],updates=updates,name='sample_fn')

        image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')
        for idx in range(n_samples):
            vis_mf, vis_sample = sample_fn()
            print(' ... plotting sample %d' % idx)
            image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(X=vis_mf,img_shape=(28, 28),tile_shape=(1, n_chains),tile_spacing=(1, 1))
            
        image = Image.fromarray(image_data)
        image.save('samples.png')
        os.chdir('../')

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    mnist = np.load('mnist_bin.npy')
    mnist_train = mnist[:50000]
    mnist_test  = mnist[50000:]

    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols

    rbm = RBM(n_visible=img_size,n_hidden=700)

    shaped_mnist_train = shape_data(mnist_train)
    rbm.train(shaped_mnist_train)

    shaped_mnist_test = shape_data(mnist_test)
    rbm.sample(shaped_mnist_test)
    # sample from rbm model
    # s = rbm.sample()