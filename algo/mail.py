import smtplib 
def send_mail(Crypto):
  smtp_server = smtplib.SMTP('smtp.gmail.com',587)
  smtp_server.ehlo()
  smtp_server.starttls()
  smtp_server.login('xxxxxx','xxxxxx')
  smtp_server.sendmail('xxxxx','xxxx',Crypto)
  smtp_server.quit()
