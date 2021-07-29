from pyngrok import ngrok
import os

# Publish Web App (Run this again whenever you make changes)
public_url = ngrok.connect(port='80')
print (public_url)
os.system("streamlit run --server.port 80 app.py")