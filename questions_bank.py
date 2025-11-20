# -------------------------------------------------------------
# questions_bank.py
# 120 Beginner-Friendly AI Questions
# -------------------------------------------------------------

QUESTION_BANK = {

    # =========================================================
    # 1. MACHINE LEARNING — 30 QUESTIONS
    # =========================================================
    "Machine Learning": [
        {
            "question": "What does ML stand for?",
            "options": ["Machine Logic", "Machine Learning", "Model Learning", "Massive Learning"],
            "answer": "Machine Learning",
            "explanation": "ML stands for Machine Learning, a field of AI that enables computers to learn patterns from data."
        },
        {
            "question": "Which of the following is a type of machine learning?",
            "options": ["Reactive Learning", "Supervised Learning", "Contextual Learning", "Sequential Learning"],
            "answer": "Supervised Learning",
            "explanation": "Supervised learning uses labeled data to train models."
        },
        {
            "question": "In ML, what is a dataset?",
            "options": ["Random numbers", "A group of files", "A collection of data samples", "A computer program"],
            "answer": "A collection of data samples",
            "explanation": "A dataset contains many examples that models learn from."
        },
        {
            "question": "Which one is an example of supervised learning?",
            "options": ["Clustering", "Linear Regression", "Dimensionality Reduction", "Auto-encoding"],
            "answer": "Linear Regression",
            "explanation": "Linear regression uses labeled data to predict continuous values."
        },
        {
            "question": "What is overfitting?",
            "options": [
                "Model performs well on new data",
                "Model learns noise instead of patterns",
                "Model is too simple",
                "Model is too slow"
            ],
            "answer": "Model learns noise instead of patterns",
            "explanation": "Overfitting happens when the model memorizes the training data instead of learning generalizable patterns."
        },
        {
            "question": "What is underfitting?",
            "options": [
                "Model is too complex",
                "Model performs exceptionally well",
                "Model cannot learn enough patterns",
                "Model uses too much memory"
            ],
            "answer": "Model cannot learn enough patterns",
            "explanation": "Underfitting means the model is too simple and fails to capture relationships in data."
        },
        {
            "question": "Which algorithm is used for classification?",
            "options": ["K-Means", "Logistic Regression", "PCA", "Apriori"],
            "answer": "Logistic Regression",
            "explanation": "Logistic regression is widely used for binary and multi-class classification."
        },
        {
            "question": "Which term refers to the error between predicted and actual values?",
            "options": ["Loss", "Gain", "Speed", "Power"],
            "answer": "Loss",
            "explanation": "Loss measures how far predictions are from actual values."
        },
        {
            "question": "What does K represent in KNN?",
            "options": ["Knowledge", "Kernel", "Number of Neighbors", "Key"],
            "answer": "Number of Neighbors",
            "explanation": "KNN predicts based on the majority of its 'K' nearest neighbors."
        },
        {
            "question": "Which ML task involves grouping similar data?",
            "options": ["Classification", "Clustering", "Regression", "Reinforcement"],
            "answer": "Clustering",
            "explanation": "Clustering finds natural groups in data without labels."
        },
        {
            "question": "Which is an example of a regression problem?",
            "options": ["Spam detection", "Digit recognition", "Predicting house prices", "Topic modeling"],
            "answer": "Predicting house prices",
            "explanation": "Regression predicts continuous numerical values."
        },
        {
            "question": "What is the purpose of splitting data into training and testing?",
            "options": [
                "To confuse the model",
                "To evaluate the model’s performance",
                "To shorten training",
                "To reduce memory usage"
            ],
            "answer": "To evaluate the model’s performance",
            "explanation": "Testing data checks how well the model generalizes to unseen data."
        },
        {
            "question": "Which of these is a performance metric for classification?",
            "options": ["Accuracy", "Mean Squared Error", "R² Score", "MAE"],
            "answer": "Accuracy",
            "explanation": "Accuracy measures how many predictions were correct."
        },
        {
            "question": "What is a feature?",
            "options": ["A column of data", "An image", "A computer part", "A random number"],
            "answer": "A column of data",
            "explanation": "Features are measurable properties of the data, often represented as columns."
        },
        {
            "question": "Which algorithm is used for clustering?",
            "options": ["Decision Trees", "K-Means", "Naive Bayes", "Linear Regression"],
            "answer": "K-Means",
            "explanation": "K-Means groups data into clusters based on similarity."
        },
        {
            "question": "What does a decision tree do?",
            "options": [
                "Clusters data",
                "Predicts by splitting data based on conditions",
                "Compresses data",
                "Encrypts text"
            ],
            "answer": "Predicts by splitting data based on conditions",
            "explanation": "Decision trees split data step-by-step to make predictions."
        },
        {
            "question": "Which one is a supervised ML algorithm?",
            "options": ["PCA", "K-Means", "Random Forest", "GANs"],
            "answer": "Random Forest",
            "explanation": "Random Forest is a supervised ensemble classifier/regressor."
        },
        {
            "question": "What is a label in ML?",
            "options": ["An ID number", "The correct output value", "A file name", "A prediction"],
            "answer": "The correct output value",
            "explanation": "Labels are the answers used in supervised training."
        },
        {
            "question": "What does ML model training mean?",
            "options": [
                "Teaching a model to memorize data",
                "Teaching a model to learn patterns",
                "Running a model fast",
                "Reducing data size"
            ],
            "answer": "Teaching a model to learn patterns",
            "explanation": "Training adjusts model parameters to learn relationships in data."
        },
        {
            "question": "Which term refers to improving model performance by combining multiple models?",
            "options": ["Ensembling", "Clustering", "Embedding", "Regularizing"],
            "answer": "Ensembling",
            "explanation": "Ensemble methods like bagging and boosting improve accuracy."
        },
        {
            "question": "Which of the following is NOT a machine learning type?",
            "options": ["Supervised", "Unsupervised", "Reinforcement", "Repetitive"],
            "answer": "Repetitive",
            "explanation": "Repetitive learning is not an ML category."
        },
        {
            "question": "Which of these is a classic ML library?",
            "options": ["NumPy", "React", "Scikit-Learn", "Photoshop"],
            "answer": "Scikit-Learn",
            "explanation": "Scikit-Learn provides many ML algorithms in Python."
        },
        {
            "question": "What is the main goal of ML?",
            "options": ["Make websites", "Teach machines to learn patterns", "Create videos", "Repair computers"],
            "answer": "Teach machines to learn patterns",
            "explanation": "The core purpose of ML is to enable systems to learn from data."
        },
        {
            "question": "Which algorithm is best for predicting YES/NO?",
            "options": ["Logistic Regression", "K-Means", "PCA", "DBSCAN"],
            "answer": "Logistic Regression",
            "explanation": "Logistic regression predicts binary categories."
        },
        {
            "question": "What is 'training data'?",
            "options": ["Data used to test models", "Data used to train models", "Incorrect data", "Unlabeled data"],
            "answer": "Data used to train models",
            "explanation": "Training data teaches the model."
        },
        {
            "question": "What does a confusion matrix measure?",
            "options": [
                "How confused the model is",
                "True and false predictions",
                "Training time",
                "Dataset size"
            ],
            "answer": "True and false predictions",
            "explanation": "It shows correct and incorrect classification counts."
        },
        {
            "question": "What is model generalization?",
            "options": [
                "Memorizing data",
                "Performing well on new data",
                "Running fast",
                "Using less memory"
            ],
            "answer": "Performing well on new data",
            "explanation": "Generalization is the ability to handle unseen samples."
        },
        {
            "question": "Which method is used to reduce overfitting?",
            "options": ["Adding noise", "Regularization", "Adding more layers", "Skipping training"],
            "answer": "Regularization",
            "explanation": "Regularization prevents models from becoming too complex."
        },
        {
            "question": "What is a test set used for?",
            "options": ["Training", "Hyperparameter tuning", "Final model evaluation", "Data cleaning"],
            "answer": "Final model evaluation",
            "explanation": "A test set measures real-world performance."
        },
    ],

      # =========================================================
    # 2. DEEP LEARNING — 30 QUESTIONS
    # =========================================================
    "Deep Learning": [
        {
            "question": "What is deep learning?",
            "options": [
                "A type of AI based on deep oceans",
                "A subset of ML using neural networks with many layers",
                "A programming technique",
                "A type of cloud storage"
            ],
            "answer": "A subset of ML using neural networks with many layers",
            "explanation": "Deep learning models use multilayer neural networks to learn complex patterns."
        },
        {
            "question": "Which structure is the basic unit of a neural network?",
            "options": ["Neuron", "Atom", "Pixel", "Block"],
            "answer": "Neuron",
            "explanation": "A neuron processes inputs and passes information forward."
        },
        {
            "question": "What is the input layer?",
            "options": [
                "The final output",
                "The first layer that receives data",
                "The layer that stores predictions",
                "A hidden layer"
            ],
            "answer": "The first layer that receives data",
            "explanation": "The input layer takes raw data into the network."
        },
        {
            "question": "What does the output layer do?",
            "options": [
                "Processes raw input",
                "Produces the final prediction",
                "Stores training history",
                "Deletes noise from data"
            ],
            "answer": "Produces the final prediction",
            "explanation": "The output layer determines the final result of a model."
        },
        {
            "question": "What does ReLU stand for?",
            "options": [
                "Relative Learning Unit",
                "Rectified Linear Unit",
                "Reactive Logic Utility",
                "Recursive Learning Utility"
            ],
            "answer": "Rectified Linear Unit",
            "explanation": "ReLU is a popular activation function in deep learning."
        },
        {
            "question": "What is an epoch?",
            "options": [
                "One complete pass through the training data",
                "A type of data",
                "The number of neurons",
                "The model size"
            ],
            "answer": "One complete pass through the training data",
            "explanation": "Training the model through all samples once is called an epoch."
        },
        {
            "question": "What is a convolutional neural network?",
            "options": [
                "A network used for text",
                "A network designed for image tasks",
                "A network for audio only",
                "A network with no layers"
            ],
            "answer": "A network designed for image tasks",
            "explanation": "CNNs are commonly used for image recognition and vision tasks."
        },
        {
            "question": "RNN stands for:",
            "options": ["Random Neural Node", "Recurrent Neural Network", "Rapid Network Node", "Recursive Neural Net"],
            "answer": "Recurrent Neural Network",
            "explanation": "RNNs handle sequential data like text or time series."
        },
        {
            "question": "What is a hidden layer?",
            "options": [
                "A secret layer",
                "Layers between input and output",
                "A layer that hides data",
                "A layer that deletes information"
            ],
            "answer": "Layers between input and output",
            "explanation": "Hidden layers learn internal representations."
        },
        {
            "question": "What is backpropagation?",
            "options": [
                "A method for sending data backwards",
                "A method for updating neural network weights",
                "A way to visualize neural networks",
                "A way to clean data"
            ],
            "answer": "A method for updating neural network weights",
            "explanation": "Backpropagation computes gradients to adjust weights."
        },
        {
            "question": "What is dropout used for?",
            "options": [
                "Speeding up CPUs",
                "Preventing overfitting",
                "Increasing training samples",
                "Improving memory"
            ],
            "answer": "Preventing overfitting",
            "explanation": "Dropout randomly disables neurons to improve generalization."
        },
        {
            "question": "Which deep learning library is popular?",
            "options": ["TensorFlow", "Excel", "React", "Premiere Pro"],
            "answer": "TensorFlow",
            "explanation": "TensorFlow is widely used for neural network training."
        },
        {
            "question": "What does GPU stand for?",
            "options": ["General Processing Unit", "Graphics Processing Unit", "Graphical Programming Utility", "General Program Unit"],
            "answer": "Graphics Processing Unit",
            "explanation": "GPUs accelerate deep learning computations."
        },
        {
            "question": "Which optimization algorithm is widely used?",
            "options": ["Adam", "Shift", "Boost", "Median"],
            "answer": "Adam",
            "explanation": "Adam optimizes networks more efficiently than basic gradient descent."
        },
        {
            "question": "Which activation function outputs values between 0 and 1?",
            "options": ["ReLU", "Sigmoid", "Tanh", "Softmax"],
            "answer": "Sigmoid",
            "explanation": "Sigmoid compresses values into the (0, 1) range."
        },
        {
            "question": "Softmax is used for:",
            "options": ["Binary classification", "Multi-class classification", "Clustering", "Regression"],
            "answer": "Multi-class classification",
            "explanation": "Softmax converts outputs into class probabilities."
        },
        {
            "question": "Which layer is common in CNNs?",
            "options": ["Pooling", "Embedding", "Flatten", "Decoder"],
            "answer": "Pooling",
            "explanation": "Pooling reduces image dimensions and helps detect features."
        },
        {
            "question": "What is a tensor?",
            "options": ["A number", "A data structure similar to arrays", "A file type", "A CPU feature"],
            "answer": "A data structure similar to arrays",
            "explanation": "Deep learning frameworks use tensors to store data."
        },
        {
            "question": "What is batch size?",
            "options": [
                "Memory size",
                "Number of samples processed before updating weights",
                "Network size",
                "Number of layers"
            ],
            "answer": "Number of samples processed before updating weights",
            "explanation": "Training happens in batches to improve stability."
        },
        {
            "question": "Which model is best for text sequences?",
            "options": ["CNN", "RNN", "GAN", "DBN"],
            "answer": "RNN",
            "explanation": "RNNs are good for sequence-based tasks."
        },
        {
            "question": "Which model powers modern AI systems?",
            "options": ["Transformers", "KNN", "SVM", "Decision Trees"],
            "answer": "Transformers",
            "explanation": "Transformers process sequences in parallel and lead to state-of-the-art results."
        },
        {
            "question": "Which network is used for image generation?",
            "options": ["GAN", "RNN", "CNN", "SVM"],
            "answer": "GAN",
            "explanation": "Generative Adversarial Networks create synthetic images."
        },
        {
            "question": "What are weights?",
            "options": ["Model parameters", "Training samples", "Layers", "Activation functions"],
            "answer": "Model parameters",
            "explanation": "Weights adjust during training to learn patterns."
        },
        {
            "question": "Which term means the model is too large?",
            "options": ["Overfitting", "Underfitting", "Overparameterization", "Regularization"],
            "answer": "Overparameterization",
            "explanation": "A model with too many parameters risks overfitting."
        },
        {
            "question": "CNNs are commonly used for:",
            "options": ["Audio", "Images", "Time series", "Databases"],
            "answer": "Images",
            "explanation": "CNNs specialize in image-related learning."
        },
        {
            "question": "What is a filter in CNNs?",
            "options": ["A camera effect", "A small matrix used to detect features", "A software tool", "A dataset"],
            "answer": "A small matrix used to detect features",
            "explanation": "Filters slide over images to detect patterns."
        },
        {
            "question": "Why are deep networks called 'deep'?",
            "options": ["They run deep underground", "Because they use many layers", "They use deep math", "They store deep data"],
            "answer": "Because they use many layers",
            "explanation": "Deep = multiple hidden layers."
        },
        {
            "question": "Which is NOT an activation function?",
            "options": ["ReLU", "Leaky ReLU", "Softmax", "ComputeMax"],
            "answer": "ComputeMax",
            "explanation": "ComputeMax is not a valid activation function."
        },
        {
            "question": "What is fine-tuning?",
            "options": [
                "Training from scratch",
                "Adjusting a pretrained model on new data",
                "Converting data",
                "Compressing model weights"
            ],
            "answer": "Adjusting a pretrained model on new data",
            "explanation": "Fine-tuning adapts pretrained networks for specific tasks."
        },
    ],

    # =========================================================
    # 3. NATURAL LANGUAGE PROCESSING — 30 QUESTIONS
    # =========================================================
    "Natural Language Processing": [
        {
            "question": "What is NLP?",
            "options": [
                "Natural Learning Program",
                "Natural Language Processing",
                "Node Language Protocol",
                "Native Logic Predictor"
            ],
            "answer": "Natural Language Processing",
            "explanation": "NLP helps computers understand human language."
        },
        {
            "question": "Which task belongs to NLP?",
            "options": ["Object detection", "Text classification", "Signal boosting", "Image generation"],
            "answer": "Text classification",
            "explanation": "NLP is used for tasks involving natural human language."
        },
        {
            "question": "What is a token?",
            "options": ["An entire book", "A piece of text such as a word", "A color", "A noise signal"],
            "answer": "A piece of text such as a word",
            "explanation": "Tokens represent small parts of text like words or subwords."
        },
        {
            "question": "What is sentiment analysis?",
            "options": [
                "Detecting text language",
                "Detecting text emotion",
                "Detecting spelling errors",
                "Translating text"
            ],
            "answer": "Detecting text emotion",
            "explanation": "Sentiment analysis classifies text as positive, negative, or neutral."
        },
        {
            "question": "Which model is common in NLP?",
            "options": ["SVM", "CNN", "Transformer", "K-Means"],
            "answer": "Transformer",
            "explanation": "Transformers power most modern NLP systems."
        },
        {
            "question": "What is stemming?",
            "options": [
                "Removing stopwords",
                "Reducing words to root forms",
                "Counting sentences",
                "Adding punctuation"
            ],
            "answer": "Reducing words to root forms",
            "explanation": "Stemming removes endings like -ing or -ed."
        },
        {
            "question": "What is a stopword?",
            "options": ["A noisy word", "A very important word", "A common word often removed", "A noun"],
            "answer": "A common word often removed",
            "explanation": "Words like 'the' or 'is' are removed in preprocessing."
        },
        {
            "question": "NER stands for:",
            "options": ["Name Entity Rule", "Natural Event Result", "Named Entity Recognition", "Number Extraction Rule"],
            "answer": "Named Entity Recognition",
            "explanation": "NER identifies entities like people or locations."
        },
        {
            "question": "Machine translation means:",
            "options": [
                "Teaching computers math",
                "Converting one language to another",
                "Generating images",
                "Sorting documents"
            ],
            "answer": "Converting one language to another",
            "explanation": "Machine translation is used in systems like Google Translate."
        },
        {
            "question": "What is a vocabulary in NLP?",
            "options": ["A grammar book", "The set of tokens a model knows", "A dictionary server", "A translation app"],
            "answer": "The set of tokens a model knows",
            "explanation": "Vocabulary defines the words or subwords used by a model."
        },
        {
            "question": "Which task is summarization?",
            "options": ["Making text longer", "Shortening text while keeping meaning", "Correcting grammar", "Adding details"],
            "answer": "Shortening text while keeping meaning",
            "explanation": "Summarization extracts key content into shorter form."
        },
        {
            "question": "Which operation converts text to numbers?",
            "options": ["Vectorization", "Illumination", "Connection", "Reconstruction"],
            "answer": "Vectorization",
            "explanation": "NLP models require numeric representations."
        },
        {
            "question": "Word embeddings represent words as:",
            "options": ["Images", "Videos", "Dense numerical vectors", "Files"],
            "answer": "Dense numerical vectors",
            "explanation": "Embeddings encode semantic meaning in vector space."
        },
        {
            "question": "BERT is a type of:",
            "options": ["Database", "Transformer model", "Operating system", "Image classifier"],
            "answer": "Transformer model",
            "explanation": "BERT is a transformer-based language model."
        },
        {
            "question": "Chatbots are an example of:",
            "options": ["Image processing", "NLP application", "Audio generation", "Data sorting"],
            "answer": "NLP application",
            "explanation": "Chatbots interact using natural language."
        },
        {
            "question": "Which task identifies parts of speech?",
            "options": ["POS tagging", "MT", "OCR", "Autoencoding"],
            "answer": "POS tagging",
            "explanation": "POS tagging labels words as nouns, verbs, etc."
        },
        {
            "question": "What is tokenization?",
            "options": ["Joining text", "Splitting text into smaller units", "Removing words", "Replacing words"],
            "answer": "Splitting text into smaller units",
            "explanation": "Tokenization breaks sentences into words or subwords."
        },
        {
            "question": "OCR is used for:",
            "options": ["Reading images of text", "Writing poems", "Predicting time", "Detecting faces"],
            "answer": "Reading images of text",
            "explanation": "OCR converts scanned text into digital text."
        },
        {
            "question": "What is language modeling?",
            "options": [
                "Making languages",
                "Predicting the next word in a sequence",
                "Building dictionaries",
                "Translating languages"
            ],
            "answer": "Predicting the next word in a sequence",
            "explanation": "Language models learn probability of word sequences."
        },
        {
            "question": "Which is NOT an NLP task?",
            "options": ["Text classification", "Speech recognition", "Image filtering", "Summarization"],
            "answer": "Image filtering",
            "explanation": "Image filtering is a computer vision task."
        },
        {
            "question": "Transformers use:",
            "options": ["Recursion", "Attention mechanism", "LOOP structures", "Compression"],
            "answer": "Attention mechanism",
            "explanation": "Attention helps models focus on relevant words."
        },
        {
            "question": "What is lemmatization?",
            "options": [
                "Removing punctuation",
                "Converting words to dictionary form",
                "Counting letters",
                "Swapping words"
            ],
            "answer": "Converting words to dictionary form",
            "explanation": "Lemma reduces words to valid dictionary roots."
        },
        {
            "question": "Which model is good for long text?",
            "options": ["Simple RNN", "LSTM", "Linear regression", "K-Means"],
            "answer": "LSTM",
            "explanation": "LSTMs manage long-term dependencies in sequences."
        },
        {
            "question": "Speech-to-text is part of:",
            "options": ["Computer vision", "NLP", "Reinforcement learning", "Sorting"],
            "answer": "NLP",
            "explanation": "Speech transcription converts audio to text."
        },
        {
            "question": "Which model family includes GPT?",
            "options": ["RNN", "Transformer", "CNN", "GAN"],
            "answer": "Transformer",
            "explanation": "GPT stands for Generative Pretrained Transformer."
        },
        {
            "question": "What is word frequency counting?",
            "options": ["TF-IDF", "Normalization", "Padding", "Pooling"],
            "answer": "TF-IDF",
            "explanation": "TF-IDF measures importance of words in documents."
        },
        {
            "question": "Which tool processes text?",
            "options": ["NLTK", "Photoshop", "Blender", "Sketch"],
            "answer": "NLTK",
            "explanation": "NLTK is a Python NLP toolkit."
        },
        {
            "question": "What is machine comprehension?",
            "options": [
                "Reading and understanding text",
                "Compressing documents",
                "Sorting files",
                "Fixing grammar"
            ],
            "answer": "Reading and understanding text",
            "explanation": "Models answer questions based on given passages."
        },
        {
            "question": "Which is text generation?",
            "options": ["Predicting stock prices", "Creating new sentences", "Detecting spam", "Finding objects"],
            "answer": "Creating new sentences",
            "explanation": "Text generation produces new text sequences."
        },
    ],

    # =========================================================
    # 4. GENERATIVE AI — 30 QUESTIONS
    # =========================================================
    "Generative AI": [
        {
            "question": "What is generative AI?",
            "options": [
                "AI that only classifies data",
                "AI that generates new content",
                "AI used for networking",
                "AI that repairs devices"
            ],
            "answer": "AI that generates new content",
            "explanation": "Generative AI produces text, images, audio, and video."
        },
        {
            "question": "Which model is widely used for image generation?",
            "options": ["GAN", "RNN", "SVM", "KNN"],
            "answer": "GAN",
            "explanation": "Generative Adversarial Networks generate realistic images."
        },
        {
            "question": "GPT stands for:",
            "options": [
                "General Purpose Transformer",
                "Generative Pretrained Transformer",
                "Global Pretrained Tool",
                "General Processing Technique"
            ],
            "answer": "Generative Pretrained Transformer",
            "explanation": "GPT is a transformer-based generative model."
        },
        {
            "question": "Which AI generates images from text?",
            "options": ["ChatGPT", "DALL·E", "Excel", "Alexa"],
            "answer": "DALL·E",
            "explanation": "DALL·E creates images from textual descriptions."
        },
        {
            "question": "What is diffusion in AI?",
            "options": [
                "Cleaning images",
                "A process for generating images by removing noise",
                "Compressing files",
                "Detecting objects"
            ],
            "answer": "A process for generating images by removing noise",
            "explanation": "Diffusion models create images step-by-step."
        },
        {
            "question": "Which is a generative model?",
            "options": ["GAN", "RNN", "Decision Tree", "SVM"],
            "answer": "GAN",
            "explanation": "GANs generate synthetic data."
        },
        {
            "question": "Which term refers to generating text?",
            "options": ["NLG", "NER", "OCR", "MT"],
            "answer": "NLG",
            "explanation": "NLG stands for Natural Language Generation."
        },
        {
            "question": "What is a discriminator in GANs?",
            "options": [
                "A model that generates data",
                "A model that evaluates real vs fake data",
                "A text translator",
                "A noise filter"
            ],
            "answer": "A model that evaluates real vs fake data",
            "explanation": "The discriminator distinguishes generated data from real samples."
        },
        {
            "question": "What is the generator in GANs?",
            "options": [
                "A model that checks quality",
                "A model that creates synthetic data",
                "A model that labels images",
                "A model that clusters data"
            ],
            "answer": "A model that creates synthetic data",
            "explanation": "The generator learns to create realistic outputs."
        },
        {
            "question": "Prompt engineering is:",
            "options": [
                "Fixing images",
                "Designing effective text prompts",
                "Repairing code",
                "Creating hardware"
            ],
            "answer": "Designing effective text prompts",
            "explanation": "Prompt engineering improves user control over generative models."
        },
        {
            "question": "Which AI can generate audio?",
            "options": ["MusicLM", "GPT-2", "SVM", "CNN"],
            "answer": "MusicLM",
            "explanation": "MusicLM generates music and sound."
        },
        {
            "question": "Which term refers to combining real and generated content?",
            "options": ["Hybrid synthesis", "Blending", "Augmentation", "MixGen"],
            "answer": "Augmentation",
            "explanation": "Data augmentation uses synthetic variations to improve training."
        },
        {
            "question": "Which is a risk of generative AI?",
            "options": ["Better accuracy", "Faster networks", "Deepfakes", "Higher resolution outputs"],
            "answer": "Deepfakes",
            "explanation": "Deepfakes can be misused to create fake videos or images."
        },
        {
            "question": "Text-to-speech is:",
            "options": ["NLP", "Computer Vision", "Generative AI", "Regression"],
            "answer": "Generative AI",
            "explanation": "TTS generates speech from text."
        },
        {
            "question": "What is fine-tuning?",
            "options": [
                "Training from scratch",
                "Adjusting pretrained models on new data",
                "Compressing data",
                "Removing noise"
            ],
            "answer": "Adjusting pretrained models on new data",
            "explanation": "Fine-tuning adapts generative models to specific tasks."
        },
        {
            "question": "Which model can generate code?",
            "options": ["Stable Diffusion", "GPT-4", "VAE", "YOLO"],
            "answer": "GPT-4",
            "explanation": "GPT-4 can generate, explain, and fix code."
        },
        {
            "question": "What is a VAE?",
            "options": [
                "Visual Application Engine",
                "Variational Autoencoder",
                "Virtual AI Entity",
                "Vector Attention Encoder"
            ],
            "answer": "Variational Autoencoder",
            "explanation": "VAEs generate new data by learning latent representations."
        },
        {
            "question": "Which term refers to the 'style' of generated content?",
            "options": ["Tone", "Flavor", "Aesthetic", "Prompt"],
            "answer": "Aesthetic",
            "explanation": "Aesthetic determines the look or tone of generated output."
        },
        {
            "question": "What does 'latent space' refer to?",
            "options": [
                "Storage memory",
                "Hidden representation learned by models",
                "Cloud server",
                "GPU memory"
            ],
            "answer": "Hidden representation learned by models",
            "explanation": "Latent space captures compressed data representations."
        },
        {
            "question": "Which tool generates art?",
            "options": ["Stable Diffusion", "SQL", "PowerPoint", "OpenCV"],
            "answer": "Stable Diffusion",
            "explanation": "Stable Diffusion generates images from text prompts."
        },
        {
            "question": "Diffusion models start with:",
            "options": ["Noise", "Text", "Vectors", "Voice"],
            "answer": "Noise",
            "explanation": "Image generation begins with noise that gets refined."
        },
        {
            "question": "ChatGPT mainly generates:",
            "options": ["Audio", "Text", "Images", "Videos"],
            "answer": "Text",
            "explanation": "GPT models are text generators."
        },
        {
            "question": "Which is a challenge of generative AI?",
            "options": ["Low accuracy", "Bias and misinformation", "No GPU usage", "Low memory"],
            "answer": "Bias and misinformation",
            "explanation": "Generative models may produce incorrect or biased content."
        },
        {
            "question": "What is reinforcement fine-tuning?",
            "options": [
                "Adding noise",
                "Training with reward-based feedback",
                "Scaling model size",
                "Cleaning data"
            ],
            "answer": "Training with reward-based feedback",
            "explanation": "RFT uses human preference signals to improve model behavior."
        },
        {
            "question": "Which chatbot is based on LLMs?",
            "options": ["Siri", "Alexa", "ChatGPT", "PowerPoint"],
            "answer": "ChatGPT",
            "explanation": "ChatGPT uses large language models to converse."
        },
        {
            "question": "Text-to-image models require:",
            "options": ["Only images", "Data pairs of images and captions", "Only audio", "Only numbers"],
            "answer": "Data pairs of images and captions",
            "explanation": "Training requires image-text alignment."
        },
        {
            "question": "Why is generative AI called 'generative'?",
            "options": ["It deletes data", "It generates new content", "It compresses content", "It stores files"],
            "answer": "It generates new content",
            "explanation": "Generative models create new data samples."
        },
        {
            "question": "Which application uses generative AI?",
            "options": ["Spam detection", "Music creation", "File compression", "Database indexing"],
            "answer": "Music creation",
            "explanation": "Generative models can create original music."
        },
        {
            "question": "Autoencoders are used for:",
            "options": ["Encoding and decoding data", "Detecting faces", "Sorting text", "Improving internet speed"],
            "answer": "Encoding and decoding data",
            "explanation": "Autoencoders compress and reconstruct data."
        },
    ]
}


