# API Setup Guide

## Getting Your OpenRouter API Key

### Step 1: Create OpenRouter Account
1. Go to [OpenRouter.ai](https://openrouter.ai)
2. Sign up for a free account
3. Verify your email address

### Step 2: Generate API Key
1. Go to [API Keys](https://openrouter.ai/keys) in your dashboard
2. Click "Create Key"
3. Give your key a descriptive name (e.g., "RAG Chatbot")
4. Copy the generated API key

### Step 3: Configure Your Application
1. Open the `.env` file in your project directory
2. Replace `your_openrouter_api_key_here` with your actual API key:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Save the file

### Step 4: Test the Connection
1. Start your application: `python app.py` or run `start.bat`
2. Open [http://localhost:5000](http://localhost:5000)
3. Check the "Status" indicator in the top-right corner
4. It should show "Connected" if the API key is working

## DeepSeek R1 Model Information

- **Model Name**: `deepseek/deepseek-r1`
- **Context Length**: ~8,000 tokens
- **Cost**: Very affordable (check OpenRouter pricing)
- **Capabilities**: Advanced reasoning, document Q&A, multi-language support

## Troubleshooting

### "API Error" Status
- ✅ Check your API key is correct in `.env`
- ✅ Ensure you have credits/usage available on OpenRouter
- ✅ Verify internet connection
- ✅ Check OpenRouter service status

### "Disconnected" Status
- ✅ Check your internet connection
- ✅ Restart the application
- ✅ Verify OpenRouter.ai is accessible

### Rate Limiting
If you encounter rate limits:
- Wait a few minutes before trying again
- Consider upgrading your OpenRouter plan
- Check your usage dashboard

## Security Notes

🔒 **Keep your API key secure:**
- Never commit `.env` file to version control
- Don't share your API key publicly
- Regenerate key if compromised

🔒 **The `.env` file is already in `.gitignore`** to prevent accidental commits.

## Cost Optimization

- DeepSeek R1 is very cost-effective
- Typical costs: ~$0.001-0.002 per query
- Monitor usage in OpenRouter dashboard
- Set spending limits if needed

---

**Need help?** Check the [OpenRouter Documentation](https://openrouter.ai/docs) for more details.
