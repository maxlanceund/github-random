package com.qs

import android.content.Context
import android.net.Uri
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView
import kotlin.math.abs

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    private val playerPool = SparseArray<ExoPlayer>()
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
        var player = playerPool.get(position)

        if (player == null) {
            val renderersFactory = DefaultRenderersFactory(context)
                .setEnableDecoderFallback(true)

            player = ExoPlayer.Builder(context, renderersFactory).build()
            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            
            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)
            
            player.volume = 0f
            player.prepare()
            
            // 强制预渲染
            player.playWhenReady = true
            player.pause()
            
            playerPool.put(position, player)
        }

        // 【绝杀1】彻底防止画框重建时黑屏：保留最后一帧残影
        holder.playerView.setKeepContentOnPlayerReset(true)
        // 【绝杀2】设置一个深灰色背景，哪怕万一没画面，也不是死黑
        holder.playerView.setShutterBackgroundColor(android.graphics.Color.DKGRAY)
        
        holder.playerView.player = player

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
        // 【修改】不要直接清空 player，让画面残留在上面。
        // 直到新的视频绑定上来，才会覆盖掉。这样就绝对不会闪黑！
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        playerPool.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
            it.volume = 0f
        }

        currentPos = position

        playerPool.get(position)?.let {
            it.volume = 1f
            it.play()
        }

        val keysToRemove = mutableListOf<Int>()
        for (i in 0 until playerPool.size()) {
            val key = playerPool.keyAt(i)
            if (abs(key - currentPos) > 3) {
                val p = playerPool.valueAt(i)
                historyStore.savePosition(videos[key].toString(), p.currentPosition)
                p.release()
                keysToRemove.add(key)
            }
        }
        keysToRemove.forEach { playerPool.remove(it) }
    }

    fun saveCurrentPosition() {
        playerPool.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until playerPool.size()) {
            val p = playerPool.valueAt(i)
            historyStore.savePosition(videos[playerPool.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        playerPool.clear()
    }
}
