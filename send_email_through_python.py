import smtplib
import os

email_address = os.environ.get("hello@gmail.com")
email_password = os.environ.get("typeYourPassword")

smtp = smtplib.SMTP_SSL('localhost', 1025)
smtp.ehlo()
smtp.starttls()  # encrypt the traffic
smtp.ehlo()
smtp.login(email_address, email_password)

subject = "See the Body"
body = "Hope to see you soon!"

message = f"Subject: {subject}\n\n{body}"

try:
    smtp.sendmail(email_address, "nagsujossh2004@gmail.com", message)
    smtp.close()
    print("Email Sent!")
except:
    print("Email not sent!")
