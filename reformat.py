def reformat_formula(formula, line_length=80):
    # Replace ** with ^ for VBA compatibility
    formula = formula.replace('**', '^')
    
    # Split the formula at operators to maintain readability
    import re
    parts = re.split(r'(\s[\+\-\*/]\s)', formula)
    
    # Initialize variables
    current_line = ''
    formatted_formula = ''
    
    # Process each part
    for part in parts:
        if len(current_line) + len(part) <= line_length:
            current_line += part
        else:
            formatted_formula += current_line.strip() + ' _\n'
            current_line = part
    
    # Add the last line
    if current_line:
        formatted_formula += current_line.strip()
    
    return formatted_formula

# Example usage
long_formula = "P * 10**5 / 4124.478823 / (T + 273.15) / (0.9999996503341386 + 0.0*T**0*P**0 + 1.5624346941144112e-06*T**1*P**0 + 0.0006142456861551616*T**0*P**1 + -7.017436144301784e-08*T**2*P**0 + -1.1696586611752336e-06*T**1*P**1 + 2.2805774858375242e-07*T**0*P**2 + 8.806008765633321e-10*T**3*P**0 + 1.0865006216613406e-09*T**2*P**1 + -3.7173626537650946e-09*T**1*P**2 + 1.0184750887635526e-10*T**0*P**3 + -2.093171032044503e-12*T**4*P**0 + -7.192747915930752e-12*T**3*P**1 + 1.557805053574199e-11*T**2*P**2 + 2.5086146793082366e-12*T**1*P**3 + -7.116163927959088e-13*T**0*P**4)"

formatted_formula = reformat_formula(long_formula)
print(formatted_formula)
