css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://ikonick.com/cdn/shop/files/ronaldo-siuu-vertical_copy_canvas_g_-_main_vertical_script_1500x1500_fbc496cd-2d72-4439-a59e-561a4b54918c.jpg?v=1740775438">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i2-prod.mirror.co.uk/article13260478.ece/ALTERNATES/s1200b/0_La-Liga-Santander-FC-Barcelona-v-Alaves.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''