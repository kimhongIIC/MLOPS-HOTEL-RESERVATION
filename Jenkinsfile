pipeline{
    agent any
    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = 'mlops-project-1-458416'
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
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
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                '''	
            }
        }

        stage('Building and pushing docker image to GCR.....'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')])
                    script{
                        echo 'Building and pushing docker image to GCR.....'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/mlops-hotel-reservation:latest .
                        docker push gcr.io/${GCP_PROJECT}/mlops-hotel-reservation:latest
                        '''
                    }


            }
        }
    }
}