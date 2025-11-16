import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey, redirectUrl, fyersId, pin } = await request.json();

    // Validate required fields
    if (!appId || !secretKey || !redirectUrl || !fyersId || !pin) {
      return NextResponse.json(
        { error: 'All fields are required' },
        { status: 400 }
      );
    }

    // Create a temporary Python script to handle authentication
    const pythonScript = `
import sys
import os
sys.path.append('C:/Code/AlgoTrading/backend/src')

from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile
import asyncio
import json

async def main():
    try:
        # Your hardcoded TOTP secret
        totp_secret = "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"

        # Get access token
        access_token = await authenticate_fyers_user(
            app_id="${appId}",
            secret_key="${secretKey}",
            redirect_uri="${redirectUrl}",
            fy_id="${fyersId}",
            pin="${pin}",
            totp_secret=totp_secret
        )

        # Get user profile
        profile = await get_user_profile(access_token, "${appId}")

        result = {
            success: True,
            access_token: access_token,
            profile: profile
        }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            success: False,
            error: str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    asyncio.run(main())
`;

    try {
      // Write the temporary Python script
      const fs = require('fs');
      const path = require('path');
      const tempScriptPath = path.join(process.cwd(), 'temp_auth_script.py');

      fs.writeFileSync(tempScriptPath, pythonScript);

      // Execute the Python script
      const { stdout, stderr } = await execAsync(`python "${tempScriptPath}"`, {
        cwd: 'C:/Code/AlgoTrading/backend',
        env: {
          ...process.env,
          PYTHONPATH: 'C:/Code/AlgoTrading/backend/src'
        }
      });

      // Clean up the temporary script
      fs.unlinkSync(tempScriptPath);

      if (stderr) {
        console.error('Python script stderr:', stderr);
        return NextResponse.json(
          { error: 'Authentication script error' },
          { status: 500 }
        );
      }

      const result = JSON.parse(stdout.trim());

      if (result.success) {
        return NextResponse.json({
          success: true,
          access_token: result.access_token,
          profile: result.profile
        });
      } else {
        return NextResponse.json(
          { error: result.error || 'Authentication failed' },
          { status: 401 }
        );
      }

    } catch (error) {
      console.error('Error executing Python script:', error);
      return NextResponse.json(
        { error: 'Failed to execute authentication' },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}