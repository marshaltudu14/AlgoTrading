import { Base64 } from 'js-base64';
import { authenticator } from 'otplib';
import axios from 'axios';
import { createHash } from 'crypto';
import { FyersCredentials, FyersUser } from '@/types/fyers';

// Fyers API URLs
const FYERS_API_BASE = 'https://api-t1.fyers.in';
const FYERS_AUTH_BASE = 'https://api-t2.fyers.in';

export class FyersAuthError extends Error {
  constructor(message: string, public code?: number) {
    super(message);
    this.name = 'FyersAuthError';
  }
}

export class FyersAuthService {
  private credentials: FyersCredentials;

  constructor() {
    this.credentials = {
      app_id: process.env.FYERS_APP_ID!,
      secret_key: process.env.FYERS_SECRET_KEY!,
      redirect_uri: process.env.FYERS_REDIRECT_URI!,
      fy_id: process.env.FYERS_USER!,
      pin: process.env.FYERS_PIN!,
      totp_secret: process.env.FYERS_TOTP!,
    };

    // Validate required environment variables
    const missingVars = Object.entries(this.credentials)
      .filter(([, value]) => !value)
      .map(([key]) => key);

    if (missingVars.length > 0) {
      throw new Error(`Missing environment variables: ${missingVars.join(', ')}`);
    }
  }

  private getEncodedString(str: string): string {
    return Base64.encode(str);
  }

  private async waitIfNeeded(): Promise<void> {
    // Wait if we're too close to TOTP expiry
    const currentSecond = new Date().getSeconds() % 30;
    if (currentSecond > 25) {
      const waitTime = 30 - currentSecond + 2; // Wait for next cycle + 2 seconds buffer
      await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
    }
  }

  async authenticate(): Promise<string> {
    try {
      console.log('Starting Fyers authentication for user:', this.credentials.fy_id);

      // Step 1: Send login OTP
      const otpResponse = await this.sendLoginOTP();

      // Step 2: Verify OTP with TOTP
      const verifyOTPResponse = await this.verifyOTP(otpResponse.request_key);

      // Step 3: Verify PIN
      const tokenResponse = await this.verifyPin(verifyOTPResponse.request_key);

      // Step 4: Get authorization code
      const authCode = await this.getAuthorizationCode(tokenResponse.access_token);

      // Step 5: Generate access token
      const accessToken = await this.generateAccessToken(authCode);

      console.log('Authentication successful');
      return accessToken;

    } catch (error) {
      console.error('Authentication failed:', error);
      throw error;
    }
  }

  private async sendLoginOTP(): Promise<{ request_key: string }> {
    const url = `${FYERS_AUTH_BASE}/vagator/v2/send_login_otp_v2`;
    const payload = {
      fy_id: this.getEncodedString(this.credentials.fy_id),
      app_id: "2"
    };

    console.log('Sending login OTP...');
    const response = await axios.post(url, payload);

    if (!response.data.request_key) {
      throw new FyersAuthError(`Failed to send OTP: ${JSON.stringify(response.data)}`);
    }

    return response.data;
  }

  private async verifyOTP(requestKey: string): Promise<{ request_key: string }> {
    await this.waitIfNeeded();

    const url = `${FYERS_AUTH_BASE}/vagator/v2/verify_otp`;
    const totpCode = authenticator.generate(this.credentials.totp_secret);

    const payload = {
      request_key: requestKey,
      otp: totpCode
    };

    console.log('Verifying OTP...');
    const response = await axios.post(url, payload);

    if (!response.data.request_key) {
      throw new FyersAuthError(`Failed to verify OTP: ${JSON.stringify(response.data)}`);
    }

    return response.data;
  }

  private async verifyPin(requestKey: string): Promise<{ access_token: string }> {
    const url = `${FYERS_AUTH_BASE}/vagator/v2/verify_pin_v2`;
    const payload = {
      request_key: requestKey,
      identity_type: "pin",
      identifier: this.getEncodedString(this.credentials.pin)
    };

    console.log('Verifying PIN...');
    const response = await axios.post(url, payload);

    if (!response.data.data?.access_token) {
      throw new FyersAuthError(`Failed to verify PIN: ${JSON.stringify(response.data)}`);
    }

    return response.data.data;
  }

  private async getAuthorizationCode(accessToken: string): Promise<string> {
    const url = `${FYERS_API_BASE}/api/v3/token`;
    const payload = {
      fyers_id: this.credentials.fy_id,
      app_id: this.credentials.app_id.slice(0, -4), // Remove last 4 characters
      redirect_uri: this.credentials.redirect_uri,
      appType: "100",
      code_challenge: "",
      state: "None",
      scope: "",
      nonce: "",
      response_type: "code",
      create_cookie: true
    };

    console.log('Getting authorization code...');
    const response = await axios.post(url, payload, {
      headers: {
        'authorization': `Bearer ${accessToken}`
      }
    });

    if (!response.data.Url) {
      throw new FyersAuthError(`Failed to get authorization URL: ${JSON.stringify(response.data)}`);
    }

    // Parse authorization code from URL
    const urlObj = new URL(response.data.Url);
    const authCode = urlObj.searchParams.get('auth_code');

    if (!authCode) {
      throw new FyersAuthError('Authorization code not found in response URL');
    }

    return authCode;
  }

  private async generateAccessToken(authCode: string): Promise<string> {
    const url = `${FYERS_API_BASE}/api/v3/validate-authcode`;
    const payload = {
      grant_type: "authorization_code",
      appIdHash: this.generateAppIdHash(),
      code: authCode
    };

    console.log('Generating access token...');
    const response = await axios.post(url, payload);

    if (!response.data.access_token) {
      throw new FyersAuthError(`Failed to generate access token: ${JSON.stringify(response.data)}`);
    }

    return response.data.access_token;
  }

  private generateAppIdHash(): string {
    const data = `${this.credentials.app_id}:${this.credentials.secret_key}`;
    return createHash('sha256').update(data).digest('hex');
  }

  async validateToken(accessToken: string): Promise<boolean> {
    try {
      const url = `${FYERS_API_BASE}/api/v3/profile`;
      const response = await axios.get(url, {
        headers: {
          'Authorization': `${this.credentials.app_id}:${accessToken}`
        }
      });

      return response.data.code === 200 && response.data.s === 'ok';
    } catch (error) {
      console.error('Token validation failed:', error);
      return false;
    }
  }

  async getUserProfile(accessToken: string): Promise<FyersUser> {
    try {
      console.log('Fetching user profile...');

      // Get profile data
      const profileResponse = await axios.get(`${FYERS_API_BASE}/api/v3/profile`, {
        headers: {
          'Authorization': `${this.credentials.app_id}:${accessToken}`
        }
      });

      if (profileResponse.data.code !== 200) {
        throw new FyersAuthError(`Failed to fetch profile: ${JSON.stringify(profileResponse.data)}`);
      }

      const profileData = profileResponse.data.data;

      // Get funds information
      const fundsResponse = await axios.get(`${FYERS_API_BASE}/api/v3/funds`, {
        headers: {
          'Authorization': `${this.credentials.app_id}:${accessToken}`
        }
      });

      let capital = 0;
      let available_balance = 0;

      if (fundsResponse.data.code === 200) {
        const fundLimit = fundsResponse.data.fund_limit || [];

        // Find specific balance types
        for (const fundItem of fundLimit) {
          if (fundItem.title === "Available Balance") {
            available_balance = fundItem.equityAmount || 0;
          } else if (fundItem.title === "Total Balance") {
            capital = fundItem.equityAmount || 0;
          }
        }

        // Fallbacks
        if (capital === 0 && fundLimit.length > 0) {
          capital = fundLimit[0].equityAmount || 0;
        }
        if (available_balance === 0) {
          available_balance = capital;
        }
      }

      return {
        name: profileData.name || "User",
        email: profileData.email_id || "",
        mobile: profileData.mobile_number || "",
        capital,
        available_balance,
        profile_data: profileData,
        funds_data: fundsResponse.data.fund_limit || []
      };

    } catch (error) {
      console.error('Failed to fetch profile:', error);
      throw new FyersAuthError(`Failed to fetch user profile: ${error}`);
    }
  }

  async refreshToken(refreshToken: string): Promise<string> {
    try {
      const url = `${FYERS_API_BASE}/api/v3/validate-refresh-token`;
      const payload = {
        grant_type: "refresh_token",
        appIdHash: this.generateAppIdHash(),
        refresh_token: refreshToken,
        pin: this.credentials.pin
      };

      const response = await axios.post(url, payload);

      if (!response.data.access_token) {
        throw new FyersAuthError(`Failed to refresh token: ${JSON.stringify(response.data)}`);
      }

      return response.data.access_token;
    } catch (error) {
      console.error('Token refresh failed:', error);
      throw new FyersAuthError(`Failed to refresh token: ${error}`);
    }
  }
}