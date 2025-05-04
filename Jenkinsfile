pipeline {
  agent any

  environment {
    VENV_DIR    = 'venv'
    GCP_PROJECT = 'mlops-project-1-458416'
    GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
  }

  stages {
    stage('Checkout') {
      steps {
        echo 'Cloning GitHub repo…'
        // scmGit is nonstandard; use the built-in git step instead
        git(
          url: 'https://github.com/kimhongIIC/MLOPS-HOTEL-RESERVATION.git',
          branch: 'main',
          credentialsId: 'github-token'
        )
      }
    }

    stage('Setup & Install') {
      steps {
        echo 'Creating venv and installing dependencies…'
        // All of these run in one shell context, 
        // so the 'activate' sticks for the pip commands that follow.
        sh '''
          python3 -m venv ${VENV_DIR}
          . ${VENV_DIR}/bin/activate
          pip install --upgrade pip --break-system-packages
          pip install --break-system-packages -e .
        '''
      }
    }

    stage('Build & Push Docker') {
      steps {
        withCredentials([file(credentialsId: 'gcp-service-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
          echo 'Building & pushing Docker image to GCR…'
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

    stage('Deploy to Cloud Run') {
      steps {
        withCredentials([file(credentialsId: 'gcp-service-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
          echo 'Deploying to Cloud Run…'
          sh '''
            export PATH=$PATH:${GCLOUD_PATH}
            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
            gcloud config set project ${GCP_PROJECT}

            gcloud run deploy mlops-hotel-reservation \
              --image=gcr.io/${GCP_PROJECT}/mlops-hotel-reservation:latest \
              --platform=managed \
              --region=us-central1 \
              --allow-unauthenticated
          '''
        }
      }
    }
  }
}
