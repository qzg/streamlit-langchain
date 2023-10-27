import streamlit_authenticator as stauth
hashed_passwords = stauth.Hasher(['astradb', 'astradb']).generate()
print (hashed_passwords)