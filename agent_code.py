class DQNAgent():
    """
    RL agent that utilizes a DQN algorithm to solve the Cart Pole environment.
    """
    
    def __init__(self, gamma=0.95, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.01, tau=0.99):
        """
        Initialize agent.
        
        Arguments:
            gamma (float): Future reward discounting factor
            epsilon (float): Starting value representing percentage of the time that the agent chooses a random action. E.g 0.75 = 75%
            epsilon_decay (float): Epsilon is multiplied by this factor after each episode.
            epsilon_min (float): Minimum value for epsilon.
            tau (float): Rate at which the target-q network is updated.
            
        Returns:
            None
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 
        self.tau = tau
        self.q_function = self.make_q_function()
        self.q_function.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=7.5e-4)) 
        self.target_q = self.make_q_function()
        self.target_q.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-4))
        self.target_q.set_weights(self.q_function.weights)
        self.max_memory = 10000
    
    
    def find_action(self, state):
        """
        Taking an action according to current policy of the agent. 
        
        Arguments:
            state (ndarray): cart position, cart velocity, pole angle, and pole angular velocity
        
        Returns:
            action (int): 0 or 1, corresponding to whether to push the cart left or right.
        """
        
        # If the random number is less than epsilon, we choose the action randomly
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 2)
            
        # Otherwise, choose the action that is the agent's current best guess for max future reward
        else:
            # Find action with best q-value
            # Find q-value estimates for each action using q-function
            q_vals = self.q_function.predict(state[np.newaxis, :], verbose=False)
            # Find best value by taking index of the largest argument
            action = np.argmax(q_vals)
    
        return action
    
    
    def take_step(self, env):
        """
        Take a step in the Cart Pole environment with the agent.
        
        Arguments:
            env: Cart Pole environment object
        
        Returns:
            old_state (ndarray): initial cart position, cart velocity, pole angle, and pole angular velocity
            action (int): 0 or 1, corresponding to whether the cart was pushed left or right
            reward (int): +1 if the episode has not yet terminated, 0 otherwise
            new_state (ndarray): new cart position, cart velocity, pole angle, and pole angular velocity
            done (bool): Whether or not episode has terminated
        """
        
        # Find action
        action = self.find_action(self.state)
        
        # Take step
        new_state, reward, done, info = env.step(action)[:4]
        old_state = self.state
        self.state = new_state
        
        return old_state, action, reward, new_state, done
    
    
    def make_q_function(self):
        """
        Create a q-function to estimate optimal total reward values for a given state-action pair.
        
        Arguments:
            None
        
        Returns:
            model: Untrained Keras sequential model. 
        """
        
        # Keras Sequential model
        # 2 outputs corresponding to left and right actions
        model = Sequential(
            [
                Dense(128, activation='relu', input_shape=(4,)),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(2, activation='linear')
            ]
        )
        
        return model
        
    
    def train(self):
        """
        Use memory of experience to train agent using gradient descent.
        
        Arguments:
            None
            
        Returns:
            None
        """
            
        # Define target = current reward + gamma * target q of next state
        
        # Target q network predictions of future reward
        future_reward = np.max(self.target_q.predict(self.memory[:, -4:], verbose=False), axis=1)
        # Accounting for steps where episode terminated (no future reward)
        future_reward = np.where(self.memory[:, 6], 0, future_reward)
        
        target_vals = self.memory[:, 5] + self.gamma * future_reward
        
        # Creating targets
        q_vals = self.q_function.predict(self.memory[:, :4], verbose=False)
        
        current_actions = self.memory[:, 4]
        # Replacing q_function predictions with target_q targets
        col1 = ((q_vals[:, 0] * current_actions) + ((1 - current_actions) * target_vals)).reshape(q_vals.shape[0], 1)
        col2 = ((q_vals[:, 1] * (1 - current_actions)) + (current_actions * target_vals)).reshape(q_vals.shape[0], 1)
        final_target = np.concatenate((col1, col2), axis=1)
        
        # Train q-function with mse against target-q
        self.q_function.fit(self.memory[:, :4], final_target, shuffle=True, batch_size=64, verbose=False)
        
        # Using Polyak averaging to soft update target-q to be tau * target q weights + (1 - tau) * q weights
        new_weights = [self.tau * i + (1 - self.tau) * j for i, j in zip(self.target_q.weights, self.q_function.weights)]
        self.target_q.set_weights(new_weights) 
        
        
    def go(self, env, num_rounds=100):
        """
        Run the agent till termination num_rounds times.
        
        Arguments:
            env: Cart Pole environment object
            num_rounds (int): Number of times we want the agent to run until termination.
            
        Returns:
            None
        """
        
        # Initializing memory and episode length record
        self.memory = np.zeros((1, 11))
        length_lst = []
        
        # Each round is a complete runthrough of the environment until termination
        for i in tqdm(range(num_rounds)):
            
            # Tracking whether or not current iteration has terminated
            complete = False
            
            # Resetting to initial state
            self.state = env.reset()[0]
            
            # Variable to keep track of number of steps taken in each round
            length = 0
            
            while not complete:
                
                # Continue taking steps until termination
                old_state, action, reward, new_state, done = self.take_step(env)
                
                # Add data to agent memory
                if self.memory.any():
                    curr_data = np.append(np.append(old_state, (action, reward, done)), new_state).reshape(1, 11)
                    self.memory = np.concatenate((self.memory, curr_data), axis=0)
                else: 
                    self.memory = (np.append(np.append(old_state, (action, reward, done)), new_state)).reshape(1, 11)
                
                # Sample memory if exceeding memory limit
                # Always keep initial 1000 steps' data
                if len(self.memory) > self.max_memory:
                    idx = np.random.randint(1000, len(self.memory), size=self.max_memory - 1000) 
                    self.memory = self.memory[np.append(np.arange(1000), idx), :]
                
                # Need > 64 for batch size
                if len(self.memory) > 64:
                    self.train()
                
                if done:
                    complete = True
                    
                length += 1
                
                if length > 550:
                    break

                
             # Decrease epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
            length_lst.append(length)
            
            if np.mean(length_lst[-10:]) >= 475:
                break
            
            if i % 10 == 0:
                print("Episode {}: \nlast 10 reward avg: {} \nepsilon: {} \nMemory size: {}".format(i, np.mean(length_lst[-10:]), self.epsilon, self.memory.shape[0]))
                               
        return length_lst
