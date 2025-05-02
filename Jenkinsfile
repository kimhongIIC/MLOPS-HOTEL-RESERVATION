pipeline{
    agent any

    stages{
        stage('cloning Githup repo to Jenkins'){
            steps{
                echo 'Cloning Github repo to Jenkins...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/kimhongIIC/MLOPS-HOTEL-RESERVATION.git']])
            }
        }
    }
}