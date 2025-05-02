pipeline{
    agent any
    environment {
        VENV_DIR = 'venv'
    }
    stages{
        stage('cloning Githup repo to Jenkins'){
            steps{
                echo 'Cloning Github repo to Jenkins...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/kimhongIIC/MLOPS-HOTEL-RESERVATION.git']])
            }
        }

        stage('Setting up our virtual environment and installing dependencies'){
            steps{
                echo 'Setting up our virtual environment and installing dependencies'
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR/bin/activate}
                    pip install --upgrade pip
                    pip install -e .
                '''	
            }
        }
    }
}