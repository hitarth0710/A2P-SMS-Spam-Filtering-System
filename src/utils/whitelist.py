import yaml


def load_whitelist(config_path):
    """
    Load whitelist configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        tuple: (whitelisted_domains, whitelisted_phrases)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        domains = config.get('whitelisted_domains', [])
        phrases = config.get('whitelisted_phrases', [])

        return domains, phrases
    except Exception as e:
        print(f"Error loading whitelist configuration: {e}")
        return [], []


def is_whitelisted(message, whitelisted_domains, whitelisted_phrases):
    """
    Check if a message is whitelisted based on trusted domains or phrases.
    
    Args:
        message (str): The message content to check
        whitelisted_domains (list): A list of trusted domains
        whitelisted_phrases (list): A list of trusted phrases
        
    Returns:
        bool: True if whitelisted, False otherwise
    """
    message_lower = message.lower()

    # Check for whitelisted domains
    for domain in whitelisted_domains:
        if domain.lower() in message_lower:
            return True

    # Check for whitelisted phrases
    for phrase in whitelisted_phrases:
        if phrase.lower() in message_lower:
            return True

    return False