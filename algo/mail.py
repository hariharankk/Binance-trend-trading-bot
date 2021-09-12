import smtplib 
def send_mail(Crypto):
  smtp_server = smtplib.SMTP('smtp.gmail.com',587)
  smtp_server.ehlo()
  smtp_server.starttls()
  smtp_server.login('hkannan084@gmail.com','vooszhdhsrefkdpg')
  smtp_server.sendmail('hkannan084@gmail.com','hkannan084@gmail.com',Crypto)
  smtp_server.quit()
