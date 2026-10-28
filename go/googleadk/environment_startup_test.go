package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"
)

func TestStartDevServerWithRetryRecovers(t *testing.T) {
	for failures := range 3 {
		t.Run(fmt.Sprint(failures), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				expected := &testsuite.DevServer{}
				startupErrors := []error{io.ErrUnexpectedEOF, context.DeadlineExceeded}
				attempts := 0
				started := time.Now()
				server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
					attempts++
					if attempts <= failures {
						return nil, startupErrors[attempts-1]
					}
					return expected, nil
				})

				require.NoError(t, err)
				require.Same(t, expected, server)
				require.Equal(t, failures+1, attempts)
				require.Equal(t, time.Duration(failures)*time.Second, time.Since(started))
			})
		})
	}
}

func TestStartDevServerWithRetryStopsAfterThreeAttempts(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		startupErrors := []error{io.ErrUnexpectedEOF, errors.New("invalid archive"), context.DeadlineExceeded}
		attempts := 0
		started := time.Now()
		server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
			attempts++
			return nil, startupErrors[attempts-1]
		})

		require.Nil(t, server)
		for _, startupErr := range startupErrors {
			require.ErrorIs(t, err, startupErr)
		}
		require.Equal(t, 3, attempts)
		require.Equal(t, 2*time.Second, time.Since(started))
	})
}

func TestStartDevServerWithRetryDoesNotRetryCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		attempts := 0
		started := time.Now()
		server, err := startDevServerWithRetry(func() (*testsuite.DevServer, error) {
			attempts++
			return nil, fmt.Errorf("server startup: %w", context.Canceled)
		})

		require.Nil(t, server)
		require.ErrorIs(t, err, context.Canceled)
		require.Equal(t, 1, attempts)
		require.Equal(t, time.Duration(0), time.Since(started))
	})
}
