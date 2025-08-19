from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .embed import encrypt, decrypt

# Create your views here.
class EncryptView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        input = request.data.get("input", None)
        operation = request.data.get("operation", None)

        if input:
            if operation == "encrypt":
                print(f"Encryption requested by {request.user.username}")
                encryption = encrypt(input)
                print(f"Returning encryption to {request.user.username}.")
                return Response({"encryption": encryption})
            
            elif operation == "decrypt":
                print(f"Decryption requested by {request.user.username}")
                decryption = decrypt(input)
                print(f"Returning decryption to {request.user.username}.")
                return Response({"decryption": decryption})
        
        else:
            return Response({"result": "Error: No input found"})