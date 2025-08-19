This repo houses an encryption/decryption API using GloVe Vectors

Setting up encryption:
1) Set up a virtual environment and pip install the requirements.txt
2) Go to https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip and download the glove.42B.300d zip.
3) Extract the zip file and copy path of the resulting txt file
4) Run setup.py with the extracted file's path as an argument. This will convert the vectors to KeyedVectors and store them as a kv file.
5) Run "python manage.py makemigrations" "python manage.py migrate" and "python manage.py createsuperuser" then follow instructions to create a new superuser.
6) In a terminal running the virtual environment, run "python manage.py runserver". This will set up the server.
7) With the server up and running, access the website's admin page (http://127.0.0.1:8000/admin/), and create a basic user.

Authentication Token:
1) With the server set up and running, post a request with the base user's username and password {"username": "user", "password": "pass"}
2) This returns the access token required to make api calls.

Encrypting:
With the auth token, a post request can be made by sending the auth token as the header, and sentence you wish to be encrypted along with the string "encrypt" as the operation as the body.

header = {"Authorization": f'Bearer {access_token}', 'Content-Type': 'application/json'}
body = {'input': 'this medicine gives me a headache and I feel sick.', 'operation': 'encrypt'}

The api will then return the sentence encrypted.

Decrypting:
To decrypt, a post request can be made by sending the encryption the api sent earlier as input and the string "decrypt" as the operation in the body.

body = {'input': encryption, 'operation': 'decrypt'}

This will return the decrypted sentence.