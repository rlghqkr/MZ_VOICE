/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme base colors
        dark: {
          50: '#1a1a2e',
          100: '#16162a',
          200: '#121226',
          300: '#0e0e1f',
          400: '#0a0a18',
          500: '#060612',
          600: '#04040d',
          700: '#020208',
          800: '#010104',
          900: '#000000',
        },
        // Neon gradient colors
        neon: {
          cyan: '#00f5ff',
          blue: '#0080ff',
          purple: '#8b5cf6',
          pink: '#ec4899',
          magenta: '#ff00ff',
        },
        // Primary gradient
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        // Emotion colors (keeping for functionality)
        emotion: {
          angry: '#ef4444',
          happy: '#22c55e',
          sad: '#3b82f6',
          neutral: '#6b7280',
          fearful: '#a855f7',
          surprised: '#f59e0b',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'neon-gradient': 'linear-gradient(135deg, #00f5ff, #0080ff, #8b5cf6, #ec4899)',
        'neon-gradient-reverse': 'linear-gradient(315deg, #00f5ff, #0080ff, #8b5cf6, #ec4899)',
        'dark-gradient': 'linear-gradient(180deg, #1a1a2e 0%, #0a0a18 100%)',
      },
      boxShadow: {
        'neon-cyan': '0 0 20px rgba(0, 245, 255, 0.5), 0 0 40px rgba(0, 245, 255, 0.3)',
        'neon-purple': '0 0 20px rgba(139, 92, 246, 0.5), 0 0 40px rgba(139, 92, 246, 0.3)',
        'neon-pink': '0 0 20px rgba(236, 72, 153, 0.5), 0 0 40px rgba(236, 72, 153, 0.3)',
        'neon-gradient': '0 0 30px rgba(0, 245, 255, 0.4), 0 0 60px rgba(139, 92, 246, 0.3)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      animation: {
        'gradient-x': 'gradient-x 3s ease infinite',
        'gradient-y': 'gradient-y 3s ease infinite',
        'gradient-xy': 'gradient-xy 3s ease infinite',
        'pulse-neon': 'pulse-neon 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'gradient-y': {
          '0%, 100%': {
            'background-size': '400% 400%',
            'background-position': 'center top'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'center center'
          }
        },
        'gradient-x': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'gradient-xy': {
          '0%, 100%': {
            'background-size': '400% 400%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'pulse-neon': {
          '0%, 100%': {
            'box-shadow': '0 0 20px rgba(0, 245, 255, 0.5), 0 0 40px rgba(139, 92, 246, 0.3)'
          },
          '50%': {
            'box-shadow': '0 0 40px rgba(0, 245, 255, 0.8), 0 0 80px rgba(139, 92, 246, 0.5)'
          }
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' }
        },
        'glow': {
          '0%': { 'box-shadow': '0 0 20px rgba(0, 245, 255, 0.5)' },
          '100%': { 'box-shadow': '0 0 40px rgba(139, 92, 246, 0.8)' }
        }
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
