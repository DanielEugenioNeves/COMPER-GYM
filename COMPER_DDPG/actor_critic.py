from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
from config.exceptions import ExceptionRunType
from config import parameters as param
import os

class BaseActorCritic(object):
    def __init__(self,model_name,task_name) -> None:
        super().__init__()
        self.model = object
        self.paramdir="./"
        self.model_name=model_name
        self.task_name = task_name
    

    def forward(self,state):
        #stateph =  tf.placeholder(dtype=tf.float32, shape=state.shape)        
        fop = self.model.predict_on_batch(state)        
        #pred = np.array
        #init_op = tf.global_variables_initializer()
        #sess.run(init_op)
        #with sess.as_default():            
           #pred = sess.run(fop,feed_dict={self.inputs:state})
        #return pred
        return fop

    def save_weights(self,paramdir):
        self.paramdir = paramdir
        os.makedirs(self.paramdir, exist_ok=True)
        self.paramdir = self.paramdir+self.model_name+'.h5'
        self.model.save_weights(self.paramdir)
        
    def load_weights(self):
        try:
            loaded= False
            print("\nMLP TRY WEIGHTS LOADING!\n")
            paramdir = self.paramsidr + '/mlp_weights.h5'
            print(paramdir)
            if(os.path.exists(paramdir)):
                print(paramdir)
                self.model.load_weights(paramdir)
                print("\nMLP WEIGHTS LOADED!\n")
                loaded = True
            return loaded
        except ValueError as isnt:
            print(type(isnt),isnt)
            raise isnt 


class Actor(BaseActorCritic):
    def __init__(self,task_name,num_states,upper_bound,num_actions) -> None:
        super().__init__(model_name="actor",task_name=task_name)
        self.model = object
        self.create(num_states,upper_bound,num_actions)
        #self.compile_model()
    
    
    def create(self,num_states,upper_bound,num_actions):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)        
        self.model = Sequential([
            layers.InputLayer(input_shape=(num_states,),name="actor_input"),
            layers.BatchNormalization(),
            layers.Dense(400, activation="relu",name="actor_dense_1"),
            layers.BatchNormalization(),
            layers.Dense(300, activation="relu",name="actor_dense_2"),
            layers.BatchNormalization(),
            layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init,name="actor_output")          
            
        ])   
    
    #def create(self,num_states,upper_bound,num_actions):
    #    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    #    inputs = layers.Input(shape=(num_states,),name="actor_input")
    #    out = layers.BatchNormalization()(inputs)
    #    out = layers.Dense(256, activation="relu",name="actor_dense_1")(out)
    #    out = layers.BatchNormalization()(out)
    #    out = layers.Dense(256, activation="relu",name="actor_dense_2")(out)
    #    out = layers.BatchNormalization()(out)
    #    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init,name="actor_output")(out)
    #    # Our upper bound is 2.0 for Pendulum.
    #    outputs = outputs * upper_bound
    #    self.model = tf.keras.Model(inputs, outputs)

    def compile_model(self):
        try:
            loaded = self.load_weights()           
            if(not loaded and self.run_type==param.RunType.TEST):
                raise ExceptionRunType(self.run_type,message="Weigths not found to run on test mode.")
        except Exception as isnt:
            print(type(isnt),isnt) 
            pass
      

class Critic(BaseActorCritic):
    def __init__(self,task_name,num_states,num_actions) -> None:
        super().__init__(model_name="critic",task_name=task_name)
        self.model = object
        self.create(num_states,num_actions)
        #self.compile_model()
    
    
    def create(self,num_states,num_actions):
        # State as input
        last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)                      
        state_input = layers.Input(shape=(num_states),name="critic_st_input")
        state_out = layers.BatchNormalization()(state_input)
        state_out = layers.Dense(400, activation="relu",name="critic_st_dense1")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(300, activation="relu",name="critic_st_dense2")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions),name="critic_act_input")       

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_input])       
        outputs = layers.Dense(1,kernel_initializer=last_init,name="critic_output")(concat)
        # Outputs single value for give state-action
        self.model = tf.keras.Model([state_input, action_input], outputs)

    
    
    
    #def create(self,num_states,num_actions):
    #    # State as input                 
    #    state_input = layers.Input(shape=(num_states),name="critic_st_input")
    #    state_out = layers.BatchNormalization()(state_input)
    #    state_out = layers.Dense(16, activation="relu",name="critic_st_dense1")(state_input)
    #    state_out = layers.BatchNormalization()(state_out)
    #    state_out = layers.Dense(32, activation="relu",name="critic_st_dense2")(state_out)

    #    # Action as input
    #    action_input = layers.Input(shape=(num_actions),name="critic_act_input")
    #    action_out = layers.Dense(32, activation="relu",name="critic_act_dense1")(action_input)

        # Both are passed through seperate layer before concatenating
    #    concat = layers.Concatenate()([state_out, action_out])

    #    out = layers.Dense(256, activation="relu",name="critic_dense2")(concat)
    #    out = layers.Dense(256, activation="relu",name="critic_dense3")(out)
    #    outputs = layers.Dense(1,name="critic_output")(out)

        # Outputs single value for give state-action
    #    self.model = tf.keras.Model([state_input, action_input], outputs)

    def compile_model(self):
        try:
            loaded = self.load_weights()
            if(not loaded):
                loaded = self.load_states()
            if(not loaded and self.run_type==param.RunType.TEST):
                raise ExceptionRunType(self.run_type,message="Weigths not found to run on test mode.")
        except Exception as isnt:
            print(type(isnt),isnt) 
            pass

    

def get_actor(task_name,num_states,upper_bound,num_actions):
    return Actor(task_name,num_states,upper_bound,num_actions)


def get_critic(task_name,num_states,num_actions):
    return Critic(task_name,num_states,num_actions)