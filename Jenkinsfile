
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/YOUR_USERNAME/mlops-iris-demo.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Retrain Model') {
            steps {
                sh '.venv/bin/python retrain.py'
            }
        }

        stage('Deploy Model') {
            steps {
                sh '''
                    cp iris_model.pkl /home/ubuntu/app/iris_model.pkl
                    sudo systemctl restart ml-api
                '''
            }
        }
    }
}
