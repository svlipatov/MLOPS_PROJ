pipeline{
    agent any

// Тут DOCKER в MAC OS

    environment {
        PATH = "/usr/local/bin:$PATH"
    }

    options{
        skipDefaultCheckout(true)
    }

    stages{

// Копирование данных проекта с github
        stage('Copy data from git'){
            steps {
                git branch: params.BRANCH_NAME, url: 'git@github.com:svlipatov/MLOPS_PROJ.git'
            }
        }

// Остановка старого контейнера DOCKER. Если контейнера нет - может выдавать ошибки
        stage('Stop old container'){
            steps {
                script {
                    sh 'docker stop prediction-app && docker rm prediction-app || true'
                    sh 'docker rmi prediction-img || true'
                }
            }
        }

// Создание образа контейнера
        stage('Build image'){
            steps {
                script {
                        sh 'docker build -t prediction-img .'

                }
            }
        }

// Запуск контейнера
        stage('Run container'){
            steps {
                script {
                    sh 'docker run -d -p 8000:8000 --restart unless-stopped --name prediction-app prediction-img'
                }
            }
        }

// Активация виртуальной среды python  и установка зависимостей
        stage('Activate venv and install requirements'){
            steps {
                script {
                        sh 'python3 -m venv venv'
                        sh 'chmod +x venv/bin/activate'
                        sh './venv/bin/activate'
                        sh 'pip3 install -r requirements.txt'

                }
            }
        }
// Запуск тестов pytest
        stage('Run tests'){
            steps {
                script {
                        sh 'python3 -m pytest -p no:warnings --alluredir allure-results'

                }
            }
        }

    }
}