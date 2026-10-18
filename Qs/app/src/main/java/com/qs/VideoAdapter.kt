package com.qs

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    // 缓存播放器
    private val playerCache = SparseArray<ExoPlayer>()
    private var currentPos = 0

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val playerView: PlayerView = view.findViewById(R.id.playerView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_video, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val uri = videos[position]
        var player = playerCache.get(position)

        if (player == null) {
            val renderersFactory = DefaultRenderersFactory(context)
                .setEnableDecoderFallback(true)
                .setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER)

            val loadControl = DefaultLoadControl.Builder()
                .setBufferDurationsMs(30000, 120000, 200, 1000)
                .build()

            player = ExoPlayer.Builder(context, renderersFactory)
                .setLoadControl(loadControl)
                .build()

            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            
            // 只预加载准备，不播放！节省底层硬件资源
            player.prepare()

            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)

            player.addListener(object : Player.Listener {
                override fun onPlayerError(error: PlaybackException) {
                    Log.e("QsPlayer", "播放失败: ${error.errorCodeName}", error)
                }
            })
            
            playerCache.put(position, player)
        }

        holder.playerView.player = player

        // 只有当前页才播放，其他页全部暂停静音
        if (position == currentPos) {
            player.volume = 1f
            player.play()
        } else {
            player.volume = 0f
            player.pause()
        }

        holder.itemView.setOnClickListener {
            if (player.isPlaying) player.pause() else player.play()
        }

        holder.itemView.setOnLongClickListener {
            onLongPress()
            true
        }
    }

    override fun onViewRecycled(holder: VH) {
        super.onViewRecycled(holder)
        val pos = holder.bindingAdapterPosition
        if (pos != RecyclerView.NO_POSITION) {
            playerCache.get(pos)?.let {
                historyStore.savePosition(videos[pos].toString(), it.currentPosition)
                // 彻底释放滑出屏幕的播放器，给手机减负
                it.release()
                playerCache.remove(pos)
            }
        }
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        playerCache.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
        }

        currentPos = position

        playerCache.get(position)?.let {
            it.volume = 1f
            it.play()
        }
    }

    fun saveCurrentPosition() {
        playerCache.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until playerCache.size()) {
            val p = playerCache.valueAt(i)
            historyStore.savePosition(videos[playerCache.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        playerCache.clear()
    }
}
