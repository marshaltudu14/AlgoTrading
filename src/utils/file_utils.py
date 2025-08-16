from src.config.settings import get_settings

def get_filename(file_type: str, symbol: str, timeframe: str = None) -> str:
    """
    Get the filename for a given file type, symbol, and timeframe.
    """
    settings = get_settings()
    file_patterns = settings.get('data_processing', {}).get('file_patterns', {})
    
    if file_type == 'features_with_timeframe' and timeframe:
        pattern = file_patterns.get('features_with_timeframe', 'features_{symbol}_{timeframe}.csv')
        return pattern.format(symbol=symbol, timeframe=timeframe)
    
    pattern = file_patterns.get(file_type, '{symbol}.csv')
    return pattern.format(symbol=symbol)
