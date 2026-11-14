package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"
)

func TestStartDevServerWithRetryRecovers(t *testing.T) {
	for _, startupErr := range []error{
		nil,
		io.ErrUnexpectedEOF,
		fmt.Errorf("failed fetching info: %w", context.DeadlineExceeded),
	} {
		t.Run(fmt.Sprint(startupErr), func(t *testing.T) {
			expected := &testsuite.DevServer{}
			attempts := 0
			server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
				attempts++
				if attempts == 1 && startupErr != nil {
					return nil, startupErr
				}
				return expected, nil
			})

			require.NoError(t, err)
			require.Same(t, expected, server)
			if startupErr == nil {
				require.Equal(t, 1, attempts)
			} else {
				require.Equal(t, 2, attempts)
			}
		})
	}
}

func TestStartDevServerWithRetryStopsAfterTwoAttempts(t *testing.T) {
	startupErrors := []error{io.ErrUnexpectedEOF, errors.New("invalid archive")}
	attempts := 0
	server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
		attempts++
		return nil, startupErrors[attempts-1]
	})

	require.Nil(t, server)
	for _, startupErr := range startupErrors {
		require.ErrorIs(t, err, startupErr)
	}
	require.Equal(t, 2, attempts)
}

func TestStartDevServerWithRetryDoesNotRepeatReadinessFailuresOrCancellation(t *testing.T) {
	for _, startupErr := range []error{
		fmt.Errorf("failed connecting after timeout, last error: %w", context.DeadlineExceeded),
		errors.New("failed connecting after timeout, last error: connection refused"),
		fmt.Errorf("server startup: %w", context.Canceled),
	} {
		t.Run(startupErr.Error(), func(t *testing.T) {
			attempts := 0
			server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
				attempts++
				return nil, startupErr
			})

			require.Nil(t, server)
			require.ErrorIs(t, err, startupErr)
			require.Equal(t, 1, attempts, "a failed readiness check must not spawn another server")
		})
	}
}
