import crypto from 'crypto';
import { FYERS_ENDPOINTS } from './constants';

export async function validateAuthCode(auth_code: string, appId: string, secretKey: string) {
  try {
    // Create SHA-256 hash of app_id:app_secret
    const appIdHash = crypto
      .createHash('sha256')
      .update(`${appId}:${secretKey}`)
      .digest('hex');

    // Exchange auth_code for access_token
    const validateResponse = await fetch(FYERS_ENDPOINTS.VALIDATE_AUTHCODE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        appIdHash: appIdHash,
        code: auth_code,
      }),
    });

    const data = await validateResponse.json();

    if (data.s !== 'ok') {
      return {
        success: false,
        error: data.message || 'Failed to validate authorization code'
      };
    }

    // Get user profile information
    const profileResponse = await fetch(FYERS_ENDPOINTS.PROFILE, {
      headers: {
        'Authorization': `${appId}:${data.access_token}`,
      },
    });

    let profileData = null;
    if (profileResponse.ok) {
      const profileResult = await profileResponse.json();
      if (profileResult.s === 'ok') {
        profileData = profileResult.data;
      }
    }

    return {
      success: true,
      access_token: data.access_token,
      refresh_token: data.refresh_token,
      profile: profileData,
      appId: appId,
    };
  } catch (error) {
    console.error('Error validating auth code:', error);
    return {
      success: false,
      error: 'Internal server error'
    };
  }
}