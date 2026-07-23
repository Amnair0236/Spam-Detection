"""
explain.py

This module provides simple explanations for why a message may have
been as Spam or Ham.
"""

def generate_explanation(message, prediction):
    """
    Generate human-readable reasons for the prediction.
    
    Parameters
    ----------
    message : str
        Original user message.
    
    prediction : str
        "Spam" or "Ham"
    
    Returns
    -------
    list
        List of explanation strings.
    """
    message_lower = message.lower()

    reasons = []

    spam_keywords = [
        "free", "win", "won", "winner", "claim", "reward", "cash", 
        "money", "offer", "urgent", "click", "call now", "limited", 
        "congratulations", "prize", "gift", "guaranteed"
    ]

    if prediction == "Spam":
        for keyword in spam_keywords:
            if keyword in message_lower:
                reasons.append(f"Contains promotional keyword: '{keyword}'")
        
        if "₹" in message or "rs" in message_lower:
            reasons.append("Contains monetary value.")
        
        if "http" in message_lower or "www" in message_lower:
            reasons.append("Contains a website link.")
        
        if "!" in message:
            reasons.append("Uses excessive punctuation.")
        
        if len(reasons) == 0:
            reasons.append("The message pattern closely matches previously learned spam messages.")
    
    else:
        reasons.append("No common spam keywords were detected.")

        reasons.append("The message resembles normal conversational text.")

        if len(message.split()) < 20:
            reasons.append("Short personal communication.")
    
    return reasons

